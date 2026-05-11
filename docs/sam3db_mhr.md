# SAM3D-body + MHR-to-SMPLH Pipeline

This document describes how to use [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) as an alternative to NLF for human pose estimation in the CARI4D preprocessing pipeline.

## Overview

The pipeline replaces Step 2 (NLF) in `scripts/demo-custom.sh`:

```
Video frames + person masks + camera intrinsics
    |
    v
SAM3D-body inference (MHR mesh per frame, 18439 vertices)
    |
    v
MHR-to-SMPLH conversion (barycentric transfer + smplfitter)
    |
    v
SMPLH parameters (poses, betas, transls) in NLF-compatible .pkl format
```

The output is a drop-in replacement for `prep/run_nlf_sepK.py` -- downstream steps (`fit_smplh_global.py`, `align_monod2hum.py`, etc.) work unchanged.

## Setup

### 1. Clone SAM 3D Body

Clone into the project root (or create a symlink):

```bash
cd /path/to/cari4d_nvlabs
git clone https://github.com/facebookresearch/sam-3d-body.git sam-3d-body
```

### 2. Download Model Checkpoints

SAM 3D Body checkpoints are hosted on Hugging Face. You need to request access first:
- [`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3) (recommended, 840M params, DINOv3-H+ backbone)
- [`facebook/sam-3d-body-vith`](https://huggingface.co/facebook/sam-3d-body-vith) (631M params, ViT-H backbone)

Download and place checkpoints at:
```
sam-3d-body/checkpoints/sam-3d-body-dinov3/
    model.ckpt
    model_config.yaml
    assets/mhr_model.pt
```

### 3. Environment

The script runs in the **cari4d** conda environment. No additional environment is needed because we bypass SAM3D's built-in detector/segmentor/FOV estimator (we provide our own bounding boxes and camera intrinsics). The key dependencies (`torch`, `smplfitter`, `smplx`, `trimesh`) are already in cari4d.

If you encounter import errors, ensure these packages are installed in cari4d:
```bash
conda activate cari4d
pip install roma pyrootutils  # only if missing
```

### 4. MHR-to-SMPLH Mapping

Download the pre-built barycentric correspondence files and place them in `data/assets/`:

```bash
mkdir -p data/assets
# Download from Hugging Face
wget https://huggingface.co/nvidia/CARI4D/resolve/main/mhr2smplh_mapping.npz -O data/assets/mhr2smplh_mapping.npz
```

Files:
- `mhr2smplh_mapping.npz` -- triangle IDs + barycentric coordinates for MHR-to-SMPLH vertex transfer

The conversion code is at `prep/mhr2smplh.py`. To rebuild the mapping for a different subject (different body shape), see `mhr2smplh/README.md`.

## Usage

### Basic Usage

```bash
conda activate cari4d

python prep/run_sam3d_sepK.py \
    -o data/cari4d-demo/videogen/nlf \
    --masks_root data/cari4d-demo/videogen/masks/ \
    --video <video_path> \
    --wild_video
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--sam3d_ckpt` | `sam-3d-body/checkpoints/.../model.ckpt` | SAM3D-body checkpoint |
| `--mhr_path` | `sam-3d-body/checkpoints/.../mhr_model.pt` | MHR model asset |
| `--mapping_dir` | `data/assets/` | Pre-built MHR-to-SMPLH mapping |
| `--chunk_size` | 16 | Frames per batch (reduce for GPU OOM) |

All arguments from `run_nlf_sepK.py` are also supported (`-o`, `--masks_root`, `--video`, `--wild_video`, `--redo`, `--index`, etc.).

### In demo-custom.sh

Uncomment the SAM3D line and comment out NLF:

```bash
# Step 2: run NLF
# python prep/run_nlf_sepK.py -o ${nlf_path} --masks_root ${masks_root} --video ${video} --wild_video

# Step 2 (alternative): run SAM3D-body instead of NLF
python prep/run_sam3d_sepK.py -o ${nlf_path} --masks_root ${masks_root} --video ${video} --wild_video
```

## Output Format

Identical to NLF output (`{video_prefix}_params.pkl`):

| Key | Shape | Description |
|-----|-------|-------------|
| `poses` | (N, K, 156) | SMPLH pose (52 joints x 3 axis-angle) |
| `betas` | (N, K, 10) | SMPLH shape parameters |
| `transls` | (N, K, 3) | 3D translation in meters |
| `frames` | list[str] | Frame identifiers |
| `gender` | str | `male` or `female` |
| `kids` | list[int] | Camera view IDs |

## How It Works

### Batch Processing

SAM3D-body's `process_one_image()` processes one image at a time. To improve throughput, we bypass it and batch multiple frames by packing them into SAM3D's "person" dimension:

- Each frame (with one person) is treated as a separate "person" in a single batch
- Uses `inference_type="body"` (body decoder only, skips hand refinement)
- Default batch size: 16 frames (adjustable via `--chunk_size`)

### MHR-to-SMPLH Conversion

1. **Camera-space vertices**: SAM3D outputs `pred_vertices` in body-local space + `pred_cam_t` for camera translation. We add them to get camera-space vertices.
2. **Barycentric transfer**: Maps 18439 MHR vertices to 6890 SMPLH-topology vertices using pre-built correspondence.
3. **SMPLH fitting**: `smplfitter` closed-form solver fits pose (156), shape (10), and translation (3) parameters.

### Failure Handling

If a frame produces invalid predictions (mean depth > 8m), the script searches +/-10 neighboring frames for a valid prediction, matching NLF's robustness pattern.

## Future Work

- **Implement the full pipeline in MHR**: ongoing work. Currently we convert MHR to SMPLH for compatibility with the existing CARI4D pipeline. A native MHR pipeline would avoid the conversion step and preserve MHR's higher mesh resolution (18439 vs 6890 vertices).
