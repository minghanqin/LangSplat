# LangSplat 
[Minghan Qin*](https://github.com/minghanqin), [Wanhua Li*†](https://li-wanhua.github.io/), [Jiawei Zhou*](https://latitudezhou.github.io/), [Haoqian Wang†](https://www.sigs.tsinghua.edu.cn/whq_en/main.htm), [Hanspeter Pfister](https://seas.harvard.edu/person/hanspeter-pfister)<br>(\* indicates equal contribution, † means Co-corresponding author)<br>| [Webpage](https://langsplat.github.io/) | [Full Paper]() | [Video](https://www.youtube.com/watch?v=XMlyjsei-Es) |<br>| [Datasets with language feature]() | [Pre-trained Models]()|<br>

![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "LangSplat: 3D Language Gaussian Splatting" (Arxiv 2024), which can be found [here](). We further provide the preprocessed datasets 3D-OVS with language feature, as well as pre-trained models. 

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{qin2023Lang,
  title={LangSplat: 3D Language Gaussian Splatting},
  author={Qin, Minghan and Li, Wanhua and Zhou, Jiawei and Wang, Haoqian and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2310.06275},
  year={2023}
}</code></pre>
  </div>
</section>

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:minghanqin/LangSplat.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/minghanqin/LangSplat.git --recursive
```

## Overview

The codebase has 3 main components:
- A PyTorch-based optimizer to produce a LangSplat model from SfM datasets with language feature inputs to
- A scene-wise language autoencode to alleviate substantial memory demands imposed by explicit modeling.
- A script to help you turn your own images into optimization-ready SfM data sets with language feature

The components have been tested on Ubuntu Linux 18.04. Instructions for setting up and running each of them are found in the sections below.

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used VS Code)
- CUDA SDK 11 for PyTorch extensions (we used 11.8)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Environment Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate langsplat
```

### QuickStart

Download the pretrained model to ```output/```, then simply use

```shell
python render.py -m output/$CASENAME --include_feature
```


## Processing your own Scenes

### Before getting started
Firstly, put your images into the data dir.
```
<dataset_name>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
```
Secondly, you need to acquire the following dataset format and a pre-trained RGB model follow the [3dgs](https://github.com/graphdeco-inria/gaussian-splatting) repository.

```
<dataset_name>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---output
|   |---<dataset_name>
|   |   |---point_cloud/iteration_30000/point_cloud.ply
|   |   |---cameras.json
|   |   |---cfg_args
|   |   |---chkpnt30000.pth
|   |   |---input.ply
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```


### Environment setup.
  Please install [segment-anything-langsplat](https://github.com/minghanqin/segment-anything-langsplat) and download the checkpoints of SAM from [here](https://github.com/facebookresearch/segment-anything) to ```ckpts/```.
### Pipeline
Follow the ```process.sh``` and train LangSplat on your own scenes.
- **Step 1: Generate Language Feature of the Scenes.**
  Put the image data into the "input" directory under the ```<dataset_name>/```, then run the following code.
  ```
  python proprecess.py --dataset_path $dataset_path 
  ```
- **Step 2: Train the Autoencoder and get the lower-dims Feature.**
  ```
  TBD
  ```

  Our model expect the following dataset structure in the source path location:
  ```
  <dataset_name>
  |---images
  |   |---<image 0>
  |   |---<image 1>
  |   |---...
  |---language_feature
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---language_feature_dim3
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---output
  |   |---<dataset_name>
  |   |   |---point_cloud/iteration_30000/point_cloud.ply
  |   |   |---cameras.json
  |   |   |---cfg_args
  |   |   |---chkpnt30000.pth
  |   |   |---input.ply
  |---sparse
      |---0
          |---cameras.bin
          |---images.bin
          |---points3D.bin
  ```
- **Step 3: Train the LangSplat.**
  ```
  python train.py -s dataset_path -m output/${casename} --start_checkpoint $dataset_path/output/$casename/chkpnt30000.pth --feature_level ${level}
  ```
- **Step 4: Render the LangSplat.**
  ```
  python render.py -s dataset_path -m output/${casename} --feature_level ${level}
  ```  
## TODO list:
- [x] release the code of the optimizer
- [ ] release the code of the autoencoder
- [x] release the code of the segment-anything-langsplat
- [ ] update the arxiv link
- [ ] release the preprocessed dataset and the pretrained model

This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.
