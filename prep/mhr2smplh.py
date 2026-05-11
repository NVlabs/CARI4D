"""Convert MHR posed vertices (from sam-3d-body) to SMPLH parameters.

Uses the barycentric correspondence map built by build_mhr2smplh_mapping.py
to transfer MHR posed meshes to SMPLH topology, then fits SMPLH parameters
via smplfitter.

This is a reusable component: copy this file + the mapping .npz files to
any project that needs MHR→SMPLH conversion.

Usage:
    python -m prep.mhr2smplh \
        --pth_file ../sam-3d-body/out/Date03_Sub03_chairwood_lift.0.pth \
        --mapping_dir debug/mhr2smplh \
        --smplh_root data/smpl/smplh \
        --output_dir debug/mhr2smplh/results
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

# Compat shims for older packages (chumpy / numpy 2.x / Python 3.12)
import inspect
for _attr in ["bool", "int", "float", "complex", "object", "str"]:
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, object))
np.unicode = str  # type: ignore
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore

import smplx
import torch
import trimesh
from smplfitter.pt import BodyModel as SmplfitBodyModel, BodyFitter as SmplfitBodyFitter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Barycentric transfer (GPU)
# ---------------------------------------------------------------------------

def transfer_via_barycentric_gpu(
    posed_verts: torch.Tensor,
    src_faces: torch.Tensor,
    tri_ids: torch.Tensor,
    bary_coords: torch.Tensor,
) -> torch.Tensor:
    """GPU batched barycentric transfer: (B, V_src, 3) → (B, V_tgt, 3)."""
    fv = posed_verts[:, src_faces[tri_ids]]  # (B, V_tgt, 3, 3)
    return (fv * bary_coords[None, :, :, None]).sum(dim=2)


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert_mhr_to_smplh(
    pth_file: str,
    mapping_dir: str,
    smplh_root: str,
    output_dir: str,
    batch_size: int = 64,
    max_frames: int = -1,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    map_path = Path(mapping_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = time.time()

    # --- Load mapping ---
    print("Loading mapping ...")
    mapping = np.load(str(map_path / "mhr2smplh_mapping.npz"))
    tri_ids = mapping["triangle_ids"]        # (6890,)
    bary_coords = mapping["baryc_coords"]    # (6890, 3)
    mhr_faces = mapping["mhr_faces"]         # (36874, 3)
    gender = str(mapping["gender"])

    shape_data = np.load(str(map_path / "smplh_shape_params.npz"))
    best_transl = torch.tensor(shape_data["transl"], dtype=torch.float32)  # (1, 3)

    print(f"  Gender: {gender}")
    print(f"  MHR faces: {mhr_faces.shape}, SMPLH target verts: {tri_ids.shape[0]}")

    # --- Load sam-3d-body output ---
    print("Loading sam-3d-body output ...")
    data = torch.load(pth_file, map_location="cpu", weights_only=False)
    n_frames = len(data)
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
    print(f"  Frames: {n_frames}")

    # Extract all pred_vertices
    all_verts = []
    for i in range(n_frames):
        v = data[i][0]["pred_vertices"]
        if isinstance(v, np.ndarray):
            v = torch.tensor(v, dtype=torch.float32)
        all_verts.append(v)
    posed_all = torch.stack(all_verts)  # (N, 18439, 3)
    print(f"  Posed vertices: {posed_all.shape}")

    # Move mapping to GPU
    faces_t = torch.tensor(mhr_faces, dtype=torch.long, device=device)
    tri_t = torch.tensor(tri_ids, dtype=torch.long, device=device)
    bary_t = torch.tensor(bary_coords, dtype=torch.float32, device=device)

    # --- Step 1: Transfer all frames ---
    print("\n" + "=" * 60)
    print("STEP 1: Barycentric transfer (MHR → SMPLH topology)")
    print("=" * 60)
    t0 = time.time()

    all_smplh_target = []
    for s in tqdm(range(0, n_frames, batch_size), desc="Transfer"):
        e = min(s + batch_size, n_frames)
        batch = posed_all[s:e].to(device)
        transferred = transfer_via_barycentric_gpu(batch, faces_t, tri_t, bary_t)
        all_smplh_target.append(transferred.cpu())

    target_smplh = torch.cat(all_smplh_target)  # (N, 6890, 3)
    t_transfer = time.time() - t0
    print(f"  Transfer: {n_frames} frames in {t_transfer:.1f}s "
          f"({t_transfer/n_frames*1000:.1f}ms/frame)")

    # Center targets (smplfitter expects origin-centered)
    target_centered = target_smplh - best_transl.unsqueeze(0)

    # --- Step 2: Fit SMPLH parameters ---
    print("\n" + "=" * 60)
    print("STEP 2: Fit SMPLH parameters (smplfitter)")
    print("=" * 60)
    t0 = time.time()

    sf_body_model = SmplfitBodyModel(
        "smplh", gender, model_root=smplh_root, num_betas=10,
    ).to(device)
    sf_fitter = SmplfitBodyFitter(sf_body_model).to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf_fitter = torch.jit.script(sf_fitter)

    fit_batch = min(batch_size * 4, 4096)
    all_rotvecs, all_betas, all_trans = [], [], []
    with torch.no_grad():
        for s in tqdm(range(0, n_frames, fit_batch), desc="SMPLH fit"):
            e = min(s + fit_batch, n_frames)
            fit_res = sf_fitter.fit(
                target_centered[s:e].to(device),
                num_iter=5,
                beta_regularizer=0.001,
                share_beta=True,
                requested_keys=["pose_rotvecs", "shape_betas", "trans"],
            )
            all_rotvecs.append(fit_res["pose_rotvecs"].cpu())
            all_betas.append(fit_res["shape_betas"][:1].cpu())
            all_trans.append(fit_res["trans"].cpu())

    shared_beta = torch.stack([b[0] for b in all_betas]).mean(0, keepdim=True)
    pose_rotvecs = torch.cat(all_rotvecs)  # (N, 156) = 52 joints × 3

    smplh_params = {
        "global_orient": pose_rotvecs[:, :3],           # (N, 3)
        "body_pose": pose_rotvecs[:, 3:66],              # (N, 63) = 21 body joints × 3
        "left_hand_pose": pose_rotvecs[:, 66:111],       # (N, 45) = 15 hand joints × 3
        "right_hand_pose": pose_rotvecs[:, 111:156],     # (N, 45) = 15 hand joints × 3
        "betas": shared_beta.expand(n_frames, -1),       # (N, 10)
        "transl": torch.cat(all_trans),                  # (N, 3)
    }
    t_fit = time.time() - t0
    print(f"  Fitting: {n_frames} frames in {t_fit:.1f}s "
          f"({t_fit/n_frames*1000:.1f}ms/frame)")

    # --- Step 3: Evaluate ---
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    smplh_model = smplx.SMPLH(
        smplh_root, gender=gender, ext="pkl",
        use_pca=False, flat_hand_mean=True,
    ).to(device)
    smplh_faces = smplh_model.faces.astype(np.int64)
    mhr_faces_np = mhr_faces.astype(np.int64)

    fitted_verts_list = []
    errors_vs_target = []
    with torch.no_grad():
        for s in range(0, n_frames, batch_size):
            e = min(s + batch_size, n_frames)
            out = smplh_model(
                betas=smplh_params["betas"][s:e].to(device),
                body_pose=smplh_params["body_pose"][s:e].to(device),
                global_orient=smplh_params["global_orient"][s:e].to(device),
                left_hand_pose=smplh_params["left_hand_pose"][s:e].to(device),
                right_hand_pose=smplh_params["right_hand_pose"][s:e].to(device),
                transl=smplh_params["transl"][s:e].to(device),
            )
            fitted_verts_list.append(out.vertices.cpu())
            tgt = target_centered[s:e].to(device)
            err = torch.sqrt(((out.vertices - tgt) ** 2).sum(-1))
            errors_vs_target.append(err.cpu())

    fitted_verts_all = torch.cat(fitted_verts_list).numpy()
    errors_cm = torch.cat(errors_vs_target).numpy() * 100

    print(f"\n  Per-vertex error vs transfer target:")
    print(f"    mean:   {errors_cm.mean():.3f}cm")
    print(f"    max:    {errors_cm.max():.3f}cm")
    print(f"    median: {np.median(errors_cm):.3f}cm")

    # Surface distance vs original MHR mesh (sample frames)
    print(f"\n  Surface distance vs original MHR (sampled frames):")
    n_eval = min(n_frames, 50)
    eval_indices = np.linspace(0, n_frames - 1, n_eval, dtype=int)
    surface_errors = []
    for i in tqdm(eval_indices, desc="Surface eval"):
        mhr_mesh = trimesh.Trimesh(
            vertices=posed_all[i].numpy(), faces=mhr_faces_np, process=False
        )
        smplh_verts_world = fitted_verts_all[i] + best_transl.numpy()[0]
        _, dists, _ = trimesh.proximity.closest_point(mhr_mesh, smplh_verts_world)
        surface_errors.append(dists)
    surface_errors = np.array(surface_errors) * 100
    print(f"    mean:   {surface_errors.mean():.3f}cm")
    print(f"    max:    {surface_errors.max():.3f}cm")
    print(f"    median: {np.median(surface_errors):.3f}cm")

    # Timing summary
    t_all = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY ({n_frames} frames)")
    print(f"  Vertex transfer:  {t_transfer:.1f}s  ({t_transfer/n_frames*1000:.1f}ms/frame)")
    print(f"  SMPLH fitting:    {t_fit:.1f}s  ({t_fit/n_frames*1000:.1f}ms/frame)")
    print(f"  Total:            {t_all:.1f}s")
    print(f"{'='*60}")

    # --- Save ---
    save = {k: v.numpy() for k, v in smplh_params.items()}
    save["errors_cm"] = errors_cm
    save["surface_errors_cm"] = surface_errors
    save["fitted_verts"] = fitted_verts_all
    save["smplh_faces"] = smplh_faces
    save["transl_offset"] = best_transl.numpy()
    save["gender"] = gender

    result_path = out_path / "smplh_results.npz"
    np.savez_compressed(str(result_path), **save)
    print(f"\nSaved results to {result_path}")

    return result_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert MHR posed vertices to SMPLH parameters"
    )
    parser.add_argument("--pth_file", required=True,
                        help="sam-3d-body .pth output file")
    parser.add_argument("--mapping_dir", required=True,
                        help="Directory with mapping from build_mhr2smplh_mapping.py")
    parser.add_argument("--smplh_root", default="data/smpl/smplh",
                        help="SMPLH model directory")
    parser.add_argument("--output_dir", default="debug/mhr2smplh/results",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Max frames to process (-1 for all)")
    args = parser.parse_args()

    convert_mhr_to_smplh(
        pth_file=args.pth_file,
        mapping_dir=args.mapping_dir,
        smplh_root=args.smplh_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
