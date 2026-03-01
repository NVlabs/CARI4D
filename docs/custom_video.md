## Processing custom video

You can find more examples from [this file](https://huggingface.co/nvidia/CARI4D/blob/main/generated-videos.zip), where we provide RGB videos, and preprocessed data which you can easily test out the pipeline `scripts/demo-custum.sh`. Example command: 
```
bash scripts/demo-custom.sh data/cari4d-demo/videogen/videos/Date03_Sub01_Suitcase_Dragging-wild.0.color.mp4
```
After you do `unzip generated-videos.zip -d data/videogen`.

For your own RGB videos, please take a look at `data/cari4d-demo/` for example data files needed to run the full CARI4D pipeline.

**Note**: our method is not designed for partially visible body or long-term occlusions. The method works usually for videos where both the person and object are mostly visible. Please checkout the teaser videos in our website and [generated videos](https://huggingface.co/nvidia/CARI4D/blob/main/generated-videos.zip) to get an idea. 


- Step 1: Use Hunyuan3D or SAM3D to obtain a reconstruction, export the mesh obj and texture files. Place the mesh under `<hy3d_root>`. We use name conversion `<seq>*_<frame_index:03d>_rgb/<seq>*_<frame_index:03d>_rgb.obj` for the object mesh file, where `frame_index` is the index to the image used to obtain the object reconstruction. Note that the mesh should have normalized scale with longest axis is normalized to [-1, 1]. We will then perform metric-scale estimation using the estimated depth from UniDepth. 

- Step 2: Prepare the human and object masks as one packed HDF5 file. Each frame should have the human mask and object mask. One can use SAM2 and GroundSAM2 to obtain the masks.

  **Output file** (example: `data/cari4d-demo/wild/masks/<seq>_masks_k<kid>.h5`):
  - HDF5 file with a top-level group named after the sequence (e.g. `Date03_Sub01_gas_wild002`)
  - Each frame contributes two datasets inside that group:
    - `<frame_id>-k<kid>.person_mask.png` — human binary mask, `bool` array, shape `(H, W)`
    - `<frame_id>-k<kid>.obj_rend_mask.png` — object binary mask, `bool` array, shape `(H, W)`, For in the wild videos, simply use `kid=0`. 
  - `<frame_id>` is a 6-digit zero-padded frame index (e.g. `000000`, `000001`, …)

- Step 3: Run openpose to detect 2D human body keypoints, and pack them into one pkl file using joblib. File format: `<seq>_GT-packed.pkl`. Example file: `data/cari4d-demo/wild/packed/Date03_Sub01_gas_wild002_GT-packed.pkl`.

  **Output file** (`<seq>_GT-packed.pkl`, saved with `joblib.dump`):

  A dict with the following keys:

  | Key | Type | Shape / value | Description |
  |-----|------|---------------|-------------|
  | `frames` | `list[str]` | `[N]` strings | 6-digit zero-padded frame indices, e.g. `['000000', '000001', …]` |
  | `joints2d` | `ndarray` | `(N, K, 25, 3)` float64 | OpenPose 2D keypoints per frame; `K` = number of views in for this sequence. For in the wild it should be `K=1`; last dim = `(x, y, confidence)` |
- Step 4: Run the full CARI4D pipeline, replace `video` in `scripts/demo-custom.sh` with your own video file.

  The video file should be an MP4 (`<seq>.0.color.mp4`) placed under `data/cari4d-demo/wild/videos/`.