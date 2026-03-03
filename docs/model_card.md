<p align="center"><b>Model Card - CARI4D</b></p>

# Overview

## Description:  
The CoCoNet model, a part of the CARI4D method, refines the initial human and object pose parameters obtained from foundation models in human and object pose estimation. It additionally predicts binary contact labels to help downstream applications. This is a transformer based model that is agnostic to specific object category. 
This model is for research and development only. 


### License/Terms of Use:   
Governing Terms: NVIDIA License. Additional Information:  https://github.com/facebookresearch/dinov2/blob/main/LICENSE.


### Deployment Geography:  
Global

### Use Case:  
Researchers and developers in the field of computer vision, VR/AR and robotics, specifically those interested in building intelligent humanoid robots, are expected to use this method for tasks such as 4D reconstruction, interaction data collection, and humanoid robot learning.

### Release Date:   
**Github:** [02/28/2026] via [https://github.com/NVlabs/CARI4D]

## Reference(s):
[CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction]  (https://arxiv.org/abs/2512.11988), Sec 3.3. 

## Model Architecture:   
**Architecture Type:** Transformers and convolutional neural networks (CNNs).
**Network Architecture:** The network contains three parts:  1) Input image encoder (DINO v2)  2) Set of blocks (CNNs and transformers) that performs matching and comparison with long-range dependencies.  3) A set of multilayer perceptions to predict updated human object poses. 

**Number of model parameters:** 194M

 
## Input:  
**Input Type(s):** Two sequences of RGB, xyz map, and human-object masks.
**Input Format(s):** Red, Green, Blue (RGB, float), xyz map (float) and masks (binary)
**Input Parameters:** The input are two sequences of images consisting of RGB, xyz map and masks. Each sequence is a 4-dimensional tensor. One sequence comes from input observation (actual RGB videos) and another one comes from synthetic renderings of the initial human-object estimations. 

**Other Properties Related to Input:** More specifically, the inputs have the following dimensions:
- RGB: 2xTx3xHxW
- xyz: 2xTx3xHxW
- masks: 2xTx2xHxW
where T is the length of this sequence and H, W are the image height and width respectively. 

## Output:  
**Output Type(s):** Human, object pose parameters, and contact scores for two hands. 
**Output Format:** float, float, float.   
**Output Parameters:** The output of this model includes the updated pose parameters and binary hand contacts for each frame in the input sequence. Each parameter is a 2D array (TxD). 
We use the [SMPL body](https://smpl.is.tue.mpg.de/) representation for the human, and object is represented using rigid rotation and translation parameters. 

**Other Properties Related to Output:** More specifically, the output parameters have the following dimensions:
- Human pose (SMPL): Tx144
- Human shape (SMPL): Tx10
- Human translation (SMPL): Tx3
- Object rotation: Tx6
- Object translation: Tx3
- Binary contact: Tx2

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.     

## Software Integration:  
The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

**Runtime Engine(s):**  
* N/A  

**Supported Hardware Microarchitecture Compatibility:**  
NVIDIA Ampere

**[Preferred/Supported] Operating System(s):**  
* Linux  


## Model Version(s): 
v1.0: Initial model version with full capabilities, unpruned and trained. 

 
## Training and Evaluation Datasets:  

## Training Dataset:
**Link:** [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/), [HODome](https://juzezhang.github.io/NeuralDome/)

**Data Modality:**  
* Image
* Video  
* Other: 3D human, object meshes and pose parameters. 
 
**Image Training Data Size**  
* 2.4 million images from the videos.  
  
 
**Video Training Data Size:**  
* 1600 videos  
 
**Non-Audio, Image, Text Training Data Size:**  
* Pose parameters corresponding to the 2.4M frames in the video.   

**Data Collection Method by dataset:**  
[Automatic/Sensors]

**Labeling Method by dataset:**  
Hybrid: Automatic/Sensors, Human.

**Properties:** The datasets capture diverse human object interaction motions using multi-view RGB or RGBD cameras. Each image is annotated with three dimensional human and object meshes with corresponding pose parameters. 


## Evaluation Dataset:
**Link:** [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/), [InterCap](https://intercap.is.tue.mpg.de/) 

**Data Collection Method by dataset:**  
[Automatic/Sensors]

**Labeling Method by dataset:**  
Hybrid: Automatic/Sensors, Human.

**Properties:** The datasets capture diverse human object interaction motions using multi-view RGB or RGBD cameras. Each image is annotated with three dimensional human and object meshes with corresponding pose parameters. 


## Inference: 
**Acceleration Engine:** Tensor(RT)

**Test Hardware:**  
* Zed Stereo Camera, 4090

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
