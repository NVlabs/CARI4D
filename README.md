# CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction
[Project Page](https://nvlabs.github.io/CARI4D/) | [ArXiv](https://arxiv.org/abs/2512.11988) 

This is the official implementation of our CVPR paper CARI4D.

Authors: Xianghui Xie, Bowen Wen, Yan Chang, Hesam Rabeti, Jiefeng Li, Ye Yuan, Gerard Pons-Moll, Stan Birchfield

<p align="left">
<img src="https://github.com/NVlabs/CARI4D/blob/website/static/videos/teaser-cut.gif" alt="teaser" width="80%"/>
</p>

## Contents
- [Installation](#installation)
- [Run demo](#run-demo)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


## Updates
- Feb 28, 2026, code released. 
- Dec 16, 2025, ArXiv released.

## TODO List
- [x] Demo on internet video. 
- [x] Demo on BEHAVE video.
- [x] Evaluation on BEHAVE dataset. 
- [ ] Example training.

## Installation

**Environment setup option 1**: Docker.  
```bash
docker pull xiexh20/cari4d && docker tag xiexh20/cari4d cari4d  # Or to build from scratch: cd docker/ && docker build --network host -t cari4d . && cd .. 
bash docker/run_container.sh
```

**Environment setup option 2**: conda (experimental)
```bash 
conda create -n cari4d python=3.10 -y 
conda activate cari4d
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt --no-build-isolation 
```

Additional files and model checkpoints:
```bash
# Clone unidepth and keep only the submodule
git clone https://github.com/lpiccinelli-eth/UniDepth.git && mv UniDepth/unidepth . && rm -rf UniDepth

# clone VolumetricSMPL and apply patches for SMPLH
git clone https://github.com/markomih/VolumetricSMPL.git && cd VolumetricSMPL && git apply ../scripts/volumetric_smplh.patch && find . -maxdepth 1 -type f -delete && mv VolumetricSMPL/*.py . && rm -r VolumetricSMPL && cd ../

# Clone NLF model weights
mkdir -p weights && wget -O weights/nlf_l_multi_0.3.2.torchscript https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript
```
Download [FoundationPosee model weights](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i) and place under the folder `weights/`. 

## Run demo

### Prepare data
**Step 1: Demo data.** Download the demo data from [here](https://huggingface.co/nvidia/CARI4D/blob/main/cari4d-demo.zip) and place them inside `data`: `unzip cari4d-demo.zip -d data`. 


**Step 2: SMPL-H model files.**

Follow the instructions from the [website](https://mano.is.tue.mpg.de/download.php), download the SMPLH pickle files and place them under `data/smpl/smplh`. It should look like this:
```bash
data/smpl
├── kid_template.npy
├── smplh
│   ├── SMPLH_female.pkl
│   ├── SMPLH_male.pkl
```
The `kid_template.npy` comes from [AGORA project](https://agora.is.tue.mpg.de/download.php), [`smpl_kid_template.npy`](https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=smpl_kid_template.npy).

**Step 3: model checkpoint.** Download pretrained [CoCoNet checkpoint](https://huggingface.co/nvidia/CARI4D/blob/main/step031397.pth) and extract to `experiments`: you should have file `experiments/cari4d-release/step031397.pth` in current folder. 

### Demo on in the wild data 
We provide the pre-processed data in the `cari4d-demo.zip` file, which corresponds to [this youtube video](https://www.youtube.com/shorts/OGtP4L_q1fI). Please download the original video your self using tools like [this](https://app.ytdown.to/en8/). And rename the downloaded file and place to path `data/cari4d-demo/wild/videos/Date03_Sub01_gas_wild002.0.color.mp4`. The video should have resolution of `608x1080` to be compatible with our pre-processed data. You can then run our demo with: 
```bash
bash scripts/demo-wild.sh
```

### Demo on BEHAVE data
The BEHAVE demo data is self-contained, you can run directly with: 
```bash
bash scripts/demo.sh
```

### Evaluation 
After running the demo BEHAVE data, you can use this command to evaluate the reconstruction: 
```bash
python tools/eval_normalize.py split_file=splits/demo-behave.json result_dir=output/opt/cari4d-release+step031397_demo-hy3d3-optv2
```
Note that you need to download the packed GT files from [here](https://huggingface.co/nvidia/CARI4D/blob/main/behave-test-gt.zip) and extract them into `output/gt/*.pth` for the evaluation.  

### Process your own video
Please see [this doc](./docs/custom_video.md) for detailed step by step instructions. 


### Reproduce results
We provide additional files to support easy reproduction of the results on BEHAVE test sequences:
- Reconstructed object meshes, download [here](https://huggingface.co/nvidia/CARI4D/blob/main/behave-recon-meshes.zip) and place them under `data/cari4d-demo/meshes`.  
- Openpose predictions, download [here](https://huggingface.co/nvidia/CARI4D/blob/main/behave-test-openpose.zip) and place them under `data/cari4d-demo/behave/packed`.
- `splits/selected-views-map.json` provides the camera view of each sequence we used to report test performance. 


## Acknowledgements
We thank Yu-Wei Chao, Umar Iqbal, Chenran Li, Daniel Zou for the helpful discussion during the project and John Welsh for the help in code release. 
This project is built on top of these amazing research projects:
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) for metri-scale depth estimation. 
- [NLF](https://github.com/isarandi/nlf) and [GENMO](https://github.com/NVlabs/GENMO) for human pose estimation. 
- [Hunyuan3D](https://github.com/Tencent/Hunyuan3D) for object mesh reconstruction. 
- [FoundationPose](https://github.com/NVlabs/FoundationPose) for object pose estimation and tracking. 


## Citation
```bibtex
@inproceedings{xie2026cari4d,
    title = {CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction},
    author = {Xie, Xianghui and Wen, Bowen and Chang, Yan and Rabeti, Hesam and Li, Jiefeng and Yuan, Ye and Pons-Moll, Gerard and Birchfield, Stan},
    booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
    month = {June},
    year = {2026},
}
```
