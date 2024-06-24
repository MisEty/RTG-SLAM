# RTG-SLAM: Real-time 3D Reconstruction at Scale Using Gaussian Splatting

Zhexi Peng, Tianjia Shao, Liu Yong, Jingke Zhou, Yin Yang, Jingdong Wang, Kun Zhou
![Teaser image](assets/teaser.png)

This repository contains the official authors implementation associated with the paper "RTG-SLAM: Real-time 3D Reconstruction
at Scale Using Gaussian Splatting", which can be found [here](https://gapszju.github.io/RTG-SLAM/static/pdfs/RTG-SLAM_arxiv.pdf).

Abstract: *We present Real-time Gaussian SLAM (RTG-SLAM), a real-time 3D reconstruction system with an RGBD camera for large-scale environments using Gaussian splatting. The system features a compact Gaussian representation and a highly efficient on-the-fly Gaussian optimization scheme. We force each Gaussian to be either opaque or nearly transparent, with the opaque ones fitting the surface and dominant colors, and transparent ones fitting residual colors. By rendering depth in a different way from color rendering, we let a single opaque Gaussian well fit a local surface region without the need of multiple overlapping Gaussians, hence largely reducing the memory and computation cost. For on-the-fly Gaussian optimization, we explicitly add Gaussians for three types of pixels per frame: newly observed, with large color errors, and with large depth errors. We also categorize all Gaussians into stable and unstable ones, where the stable Gaussians are expected to well fit previously observed RGBD images and otherwise unstable. We only optimize the unstable Gaussians and only render the pixels occupied by unstable Gaussians. In this way, both the number of Gaussians to be optimized and pixels to be rendered are largely reduced, and the optimization can be done in real time. We show real-time reconstructions of a variety of large scenes. Compared with the state-of-the-art NeRF-based RGBD SLAM, our system achieves comparable high-quality reconstruction but with around twice the speed and half the memory cost, and shows superior performance in the realism of novel view synthesis and camera tracking accuracy.*


## 1. Installation

### 1.1 Clone the Repository

```
git clone --recursive https://github.com/MisEty/RTG-SLAM.git
```

### 1.2 Python Environment
RTG-SLAM has been tested on python 3.9, CUDA=11.7, pytorch=1.13.1. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda env create -f environment.yaml
```

### 1.3 Modified ORB-SLAM2 Python Binding
We have made some changes on ORB-SLAM2 to work with our ICP front-end and you can run this script to install pangolin, opencv, orbslam and boost-python binding. 

```bash
bash build_orb.sh
```

If you encounted the problem during install pangolin:

```bash
xxx/Pangolin/src/video/drivers/ffmpeg.cpp: In function ‘std::__cxx11::string pangolin::FfmpegFmtToString(AVPixelFormat)’:
xxx/Pangolin/src/video/drivers/ffmpeg.cpp:41:41: error: ‘AV_PIX_FMT_XVMC_MPEG2_MC’ was not declared in this scope
```

You can follow this [solution](https://github.com/stevenlovegrove/Pangolin/pull/318/files?diff=split&w=0).

#### Note
For real data, backend optimization based on ORB-SLAM2 is crucial. Therefore, you need to install the python binding for ORB-SLAM2 according to the steps. We have modified some code based on [ORB_SLAM2-PythonBindings
](https://github.com/jskinn/ORB_SLAM2-PythonBindings). If you encounter any problem related to compilation, you can refer to [ORB_SLAM2-PythonBindings
](https://github.com/jskinn/ORB_SLAM2-PythonBindings) to find solutions. Our ICP front-end works well when the depth is accurate so if you only want to test on synthetic dataset like Replica, you don't need to install ORB-SLAM2 python binding.


### 1.4 Test ORB-SLAM2 Python Binding
```bash
cd thirdParty/pybind/examples
python orbslam_rgbd_tum.py # please set voc_path, association_path ...
python eval_ate.py path_to_groundtruth.txt trajectory.txt --plot PLOT --verbose
```
If the code runs without any error and the trajetory is corret, you can move on to the next step.

## 2. Dataset Preparation
### 2.1 Replica
```
bash scripts/download_replica.sh
```
### 2.2 TUM-RGBD
```bash
bash scripts/download_tum.sh
```
And copy config file to data folder.
```bash
cp configs/tum/dataset/fr1_desk.yaml data/TUM_RGBD/rgbd_dataset_freiburg1_desk/config.yaml
cp configs/tum/dataset/fr2_xyz.yaml data/TUM_RGBD/rgbd_dataset_freiburg2_xyz/config.yaml
cp configs/tum/dataset/fr3_office.yaml data/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household/config.yaml
```


### 2.3 ScanNet++
Please follow [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) to download dataset. And run
```bash
python scripts/parse_scannetpp.py --data_path scannetpp_path/download/data/8b5caf3398 --output_path data/ScanNetpp/8b5caf3398
```
### 2.4 Ours
You can donwload our dataset from [google cloud](https://drive.google.com/drive/folders/161QHjVTHRCED9WmRWAlOEhJQ_GXxgtn5?usp=sharing). 


### 2.5 Dataset

```
|-- data
    |-- Ours
        |-- hotel
    |-- Replica
        |-- office0
            |-- results
            |-- office0.ply
            |-- traj.txt
        |-- cam_params.json
    |-- TUM_RGBD
        |-- rgbd_dataset_freiburg1_desk
    |-- ScanNetpp
        |-- 8b5caf3398
            |-- color
            |-- depth
            |-- intrinsic
            |-- pose
            |-- mesh_aligned_cull.ply
```

## 3. Run
### 3.1 Replica
```bash
# Single Process: Recommended, More Stable
python slam.py --config ./configs/replica/office0.yaml
# Multi Process: 
python slam_mp.py --config ./configs/replica/office0.yaml
```

### 3.2 TUM-RGBD
```bash
# Single Process: Recommended, More Stable
python slam.py --config ./configs/tum/fr1_desk.yaml
# Multi Process: 
python slam_mp.py --config ./configs/tum/fr1_desk.yaml
```

### 3.3 ScanNet++
```bash
# Single Process: Recommended, More Stable
python slam.py --config ./configs/scannetpp/8b5caf3398.yaml
# Multi Process: 
python slam_mp.py --config ./configs/scannetpp/8b5caf3398.yaml
```

### 3.4 Ours
```bash
# Single Process: Recommended, More Stable
python slam.py --config ./configs/ours/hotel.yaml
# Multi Process: 
python slam_mp.py --config ./configs/ours/hotel.yaml
```

## 4. Evaluate
You can run metric.py to evaluate the rendering quality on Replica, ScanNet++ and Ours dataset and calculate geometry accuracy on Replica and ScanNet++.
There will be a csv result file in model path.
The tracking accuracy is estimated right after running slam.py. The ate result is in model_path/save_traj.
#### Note
The script selects all images when computing psnr, lpips and ssim. Our method adds Gaussian according to the depth so the performance may decrease in the presence of significant depth noise or invalid depth (such as transparent materials, highly reflective materials, etc.). For fairness, when evaluating novel view synthesis on ScanNet++ in the paper, we manually removed images with large invalid depth areas.


```bash
python metric.py --config config_path \
    # eval the first k frames
    ----load_frame k \
    # save pictures
    --save_pic
```

### 4.1 Replica
```bash
python metric.py --config ./configs/replica/office0.yaml
```
### 4.2 TUM-RGBD
```bash
python metric.py --config ./configs/tum/fr1_desk.yaml
```
### 4.3 ScanNet++
```bash
python metric.py --config ./configs/scannetpp/8b5caf3398.yaml # all novel view images
```
### 4.4 Ours
```bash
python metric.py --config ./configs/ours/hotel.yaml
```

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{peng2024rtgslam,
        author    = {Zhexi Peng and Tianjia Shao and Liu Yong and Jingke Zhou and Yin Yang and Jingdong Wang and Kun Zhou},
        title     = {RTG-SLAM: Real-time 3D Reconstruction at Scale using Gaussian Splatting},
        booktitle  = {ACM SIGGRAPH Conference Proceedings, Denver, CO, United States, July 28 - August 1, 2024},
        year      = {2024},
      }</code></pre>
  </div>
</section>


## Acknowledgments
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The ORB-SLAM backend is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). The ORB-SLAM2 Python Binding is based on [ORB_SLAM2-PythonBindings](https://github.com/jskinn/ORB_SLAM2-PythonBindings). The evaluation script is adopted from [NICE-SLAM](https://github.com/cvg/nice-slam) and [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM). We thank all the authors for their great work.
