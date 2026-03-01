video="data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4"
video_prefix=$(basename "$video" | cut -d. -f1)
echo $video_prefix

set -e

# Step 1: run Unidepth estimation
python prep/unidepth_behave.py --wild_video --video ${video} -o data/cari4d-demo/wild/videos/

# Step 2: run GENMO
# see: https://github.com/NVlabs/GENMO

# Step 3: align Unidepth to GENMO human
python prep/align_monod2hum.py --wild_video --nlf_path data/cari4d-demo/wild/genmo \
--masks_root data/cari4d-demo/wild/masks/ \
--video ${video}

# Step 4: run FP in tracking mode
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
--masks_root data/cari4d-demo/wild/masks/ --hy3d_root=data/cari4d-demo/meshes \
--video ${video} -o data/cari4d-demo/wild/fp-hy3d3-track

# Step 5: run CoCoNet to refine human + object
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=False data_name=test-only \
hy3d_meshes_root=data/cari4d-demo/meshes \
masks_root=data/cari4d-demo/wild/masks/ \
fp_root=data/cari4d-demo/wild/fp-hy3d3-track \
nlf_root=data/cari4d-demo/wild/genmo \
video=${video}  cam_id=0 wild_video=True \
outpath=output/coconet

# Step 6: run joint optimization
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=optv2 batch_size=192 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.006 opt_smpl_trans=False opt_betas=False  \
pth_file=output/coconet/cari4d-release+step031397_demo/${video_prefix}.pth  wild_video=True \
video_root=data/cari4d-demo/wild/videos/ \
packed_root=data/cari4d-demo/wild/packed \
masks_root=data/cari4d-demo/wild/masks/  \
hy3d_meshes_root=data/cari4d-demo/meshes outpath=output/opt
# Note: reduce batch_size if encounter GPU OOM.