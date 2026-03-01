# Run CARI4D on custom videos.
# Please follow ./docs/custom_videos.md to prepare data before running this script.
# Required data before running: 1). Object mesh. 2). Masks of human and object. 3). openpose detections of the human. 

video=$1
video_prefix=$(basename "$video" | cut -d. -f1)
video_dir=$(dirname "$video")
echo $video_prefix

# Paths that store preprocessed data:
masks_root=data/cari4d-demo/videogen/masks/ # store the masks of human and object.
packed_root=data/cari4d-demo/videogen/packed/ # store the openpose detections for each frame.
hy3d_root=data/cari4d-demo/videogen/meshes # store the reconstructed object mesh in normalized scale. 

# Paths for intermediate results: 
nlf_path=data/cari4d-demo/videogen/nlf
fp_root=data/cari4d-demo/videogen/fp-hy3d3-track
coconet_out=output/coconet

set -e

# Step 1: run Unidepth estimation
python prep/unidepth_behave.py --wild_video --video ${video} -o ${video_dir}

# Step 2: run NLF
python prep/run_nlf_sepK.py -o ${nlf_path} --masks_root ${masks_root} --video ${video} --wild_video 

# Step 3: run SMPLH fitting to get globally consistent human pose and translation
python prep/fit_smplh_global.py --wild_video --video ${video} --packed_root ${packed_root} --masks_root ${masks_root} \
    --nlf_path=${nlf_path}

# Step 4: align Unidepth to GENMO human
python prep/align_monod2hum.py --wild_video --nlf_path ${nlf_path}-opt \
--masks_root ${masks_root} \
--video ${video}

# Update the video path, pointing to the new video with aligned depth. 
video=${video_dir}-aligned/${video_prefix}.0.color.mp4 

# Step 5: estimate metric scale of the object 
python tools/estimate_scale_video.py --wild_video --video ${video} --masks_root ${masks_root} --hy3d_root ${hy3d_root} -o ${hy3d_root}-metric


# Step 5: run FP in tracking mode
python prep/fp_hy3d_track.py --viz_path x --wild_video --kid 0 \
--masks_root ${masks_root} --hy3d_root=${hy3d_root}-metric \
--video ${video} -o ${fp_root}

# Step 6: run CoCoNet to refine human + object
python run_horefine.py config=learning/configs/cari4d-release.yml split_file=splits/demo-behave.json \
use_sel_view=True render_video=True identifier=_demo use_intermediate=False data_name=test-only \
hy3d_meshes_root=${hy3d_root}-metric \
masks_root=${masks_root} \
fp_root=${fp_root} \
nlf_root=${nlf_path}-opt \
video=${video}  cam_id=0 wild_video=True \
outpath=${coconet_out}

# Step 7: run joint optimization
python learning/training/opt_refineout.py num_steps=3000 w_acc_v=600 w_contact=300  save_name=optv2 batch_size=64 opt_rot=True \
opt_trans=True w_temp=1000 w_sil=0.002 w_contact=200.0 w_pen=2.0 w_j2d=0.006 opt_smpl_trans=False opt_betas=False  \
pth_file=${coconet_out}/cari4d-release+step031397_demo/${video_prefix}.pth  wild_video=True use_input=True \
video_root=$(dirname "$video") \
packed_root=${packed_root} \
masks_root=${masks_root}  \
hy3d_meshes_root=${hy3d_root}-metric outpath=output/opt
# Note: reduce batch_size if encounter GPU OOM.