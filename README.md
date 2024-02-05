# CMTCoop - Cross Modal Transformers for Cooperative perception

This work is based on the work from ["Cross Modal Transformer: Towards Fast and Robust 3D Object Detection"](https://arxiv.org/pdf/2301.01283.pdf)

## Introduction

<!-- https://user-images.githubusercontent.com/18145538/210828888-a944817a-858f-45ef-8abc-068adeda413f.mp4

<div align="center">
  <img src="figs/cmt_eva.png" width="900" />

  <em> Performance comparison and Robustness under sensor failure. All statistics are measured on a single
Tesla A100 GPU using the best model of official repositories. All models use [spconv](https://github.com/traveller59/spconv) Voxelization module. </em>
</div><br/> -->

CMT is a transformer-based robust 3D detector for end-to-end 3D multi-modal detection. This model is extended to cooperative perception in CMTCoop to perform deep multi-model multi-view feature fusion for 3D object detection.
Through extensive, studies this work shows that the proposed model provides a mAP of 97.3% on multi-modal cooperative fusion (+6.2% increase over vehicular perception) and 96.7% on LiDAR only cooperative perception (CMTCoop-L) which runs at near-real time FPS, and a 2.1% performance gain over the current SoTA, BEVFusionCoop.

<br>
<div align="center">
  <img src="figs/CMTCoop.drawio.png" width="900" />
</div>

<!-- A DETR-like framework is designed for multi-modal detection(CMT) and lidar-only detection(CMT-L), which obtains **74.1%**(**SoTA without TTA/model ensemble**) and **70.1%** NDS separately on nuScenes benchmark. -->
<!-- Without explicit view transformation, CMT takes the image and point clouds tokens as inputs and directly outputs accurate 3D bounding boxes. CMT can be a strong baseline for further research. -->

## Preparation

### Docker installation

Docker provides an easy way to deal with package dependencies. Use the [Dockerfile](./Dockerfile) provided to build the image.

```bash
docker build . -t cmt-coop
```

Then run the image with the following command

```bash
nvidia-docker run -it --rm \
    --ipc=host --gpus all \
    -v <Path_to_datasets>:/mnt/datasets \
    -v <Path_to_pretrained_models>:/home/pretrained \
    --name cmt-coop \
    cmt-coop bash
```

<!-- ```bash
nvidia-docker run -it -v `pwd`/../data/tumtraf_i:/home/data/tumtraf_i -v <PATH_TO_COOPDET3D>:/home/coopdet3d --shm-size 16g coopdet3d /bin/bash
``` -->

### Manual Installation

Create an new environment with Anaconda or venv if required

```bash
conda create -n cmt-coop
conda activate cmt-coop
```

Install the following packages

- Python == 3.8 
- CUDA == 11.1 
- pytorch == 1.9.1 
- mmcv-full == 1.6.2 
- mmdet == 2.28.2 
- mmsegmentation == 0.30.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d) == 1.0.0rc6
- spconv-cu111 == 2.1.21 
- [flash-attn](https://github.com/HazyResearch/flash-attention) == 0.2.2
- [pypcd](https://github.com/klintan/pypcd.git) 
- open3d

Note that the repository was tested on the above versions, but may also work with later versions.

## Dataset

Follow the [mmdet3d](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md) to process the nuScenes dataset. This is only required to repeat tests on the CMT model. 

The dataset links will be released soon.

<!-- Download the TUMTraf datasets from the following links:
- [TUMTraf intersection dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_2).
- [TUMTraf Cooperative Dataset 500 frames (v0.9)](https://syncandshare.lrz.de/getlink/fiXdMni7DP5bqSsYgSpeLc/traffix-0.9)
- [TUMTraf Cooperative Dataset 800 frames (v1.0)](https://syncandshare.lrz.de/getlink/fi4gZzFh8BUn6Sw4ZC8u49/traffix-1.0).  -->

Download the [TUMTraf Dataset Development Kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit) and follow the instructions to split the TUMTraf intersection dataset into train and val sets.The TUMTraf cooperative dataset is already split into train and val sets.

```
${Root}
└── datasets
    ├── tumtraf_intersection_dataset
    |    └── train
    |    └── val
    └── tumtraf_cooperative_dataset
         └── train
         └── val
```

Finally ensure that the dataset folder has been soft linked to the `CMTCoop/data` folder.

```bash
ln -s /path_to_data_folder CMTCoop/data
```

### Data preparation

The TUMTraf dataset must be converted from Openlabel format to be compatible with mmdet3D framework

#### TUMTraf Intersection Dataset

Run this script for data preparation:

```bash
python ./tools/create_data.py a9_nusc \\
--root-path /home/CMTCoop/data/tumtraf_intersection_dataset \\
--out-dir /home/CMTCoop/data/tumtraf_intersection_processed \\
--splits training,validation
```

After data preparation, you will be able to see the following directory structure:

```
├── data
│   ├── tumtraf_intersection_dataset
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_intersection_processed
│   │   ├── a9_nusc_gt_database
|   |   ├── train
|   |   ├── val
│   │   ├── a9_nusc_infos_train.pkl
│   │   ├── a9_nusc_infos_val.pkl
│   │   ├── a9_nusc_dbinfos_train.pkl

```

#### TraffiX Cooperative Dataset

Run this script for data preparation:

```bash
python ./tools/create_data.py a9coop_nusc \\
--root-path /home/CMTCoop/data/tumtraf_cooperative_dataset \\
--out-dir /home/CMTCoop/data/tumtraf_cooperative_processed \\
--splits training,validation
```

After data preparation, you will be able to see the following directory structure:

```
├── data
│   ├── tumtraf_cooperative_dataset
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_cooperative_processed
│   │   ├── a9_nusc_coop_gt_database
|   |   ├── train
|   |   ├── val
│   │   ├── a9_nusc_coop_infos_train.pkl
│   │   ├── a9_nusc_coop_infos_val.pkl
│   │   ├── a9_nusc_coop_dbinfos_train.pkl

```


<!-- Note that the version with 500 frames (v0.9) is provided to reproduce the results listed below and in the paper as they were achieved using this older version of the dataset. The version with 800 frames (v1.0) is the latest one and includes more frames and corrected labels. -->


<!-- * Data   

PKLs and image pretrain weights are available at [Google Drive](https://drive.google.com/drive/folders/1wTdG7oG-l-nMa_400jBwJk4mEQmA_xl3?usp=sharing). -->

## Train & inference
```bash
# train
bash tools/dist_train.sh /path_to_your_config 8
# inference
bash tools/dist_test.sh /path_to_your_config /path_to_your_pth 8 --eval bbox
```

## Main Results

Results on the TUMTraf cooperative **validation set**. The FPS is evaluated on a single RTX3080 GPU.

### Evaluation Results (mAP<sub>BEV</sub> and mAP<sub>3D</sub> ) of CMTCoop on TraffiX Cooperative Dataset Test Set in South 2 FOV

| Domain        | Modality    | mAP<sub>BEV</sub> | mAP<sub>3D</sub> Easy | mAP<sub>3D</sub> Mod. | mAP<sub>3D</sub> Hard | mAP<sub>3D</sub> Avg. |
|--------------|-------------|----------------|----------------|--------------|--------------|--------------|
| Vehicle | Camera | 69.76 | 68.76 | 79.85 | 66.44 | 69.30 |
| Vehicle | LiDAR | 88.17 | 87.94 | 88.53 | 71.99 | 84.72 |
| Vehicle | Cam+LiDAR | 91.65 | 84.83 | 91.32 | 72.18 | 85.57 |
| Infra. | Camera | 71.89 | 70.86 | 80.38 | 58.72 | 71.66 |
| Infra. | LiDAR | 94.42 | 91.28 | 95.60 | 77.48 | 91.89 |
| Infra. | Camera + LiDAR | 96.09 | 91.94 | 95.15 | **82.35** | 92.16 |
| Coop. | Camera | 84.07 | 81.03 | 90.05 | 77.94 | 83.43 |
| Coop. | LiDAR | 96.68 | 92.18 | 96.77 | 82.20 | 93.43 |
| Coop. | Camera + LiDAR | **97.31** | **93.70** | **96.65** | 79.84 | **94.10** |

<!-- 7.7 10528 MiB 3951 MiB
Infra. LiDAR      17.0 5392 MiB 2175 MiB
Infra. Cam+LiDAR      5.8 11535 MiB 4067 MiB
Coop. Camera      5.6 22358 MiB 4523 MiB
Coop. LiDAR      9.8 7352MiB 2293 MiB
Coop. Cam+LiDAR      -->

### Evaluation Results of Infrastructure-only models on TUMTraf Intersection Dataset Test Set

| Model        | FOV    | Modality | mAP<sub>3D</sub> Easy | mAP<sub>3D</sub> Mod. | mAP<sub>3D</sub> Hard | mAP<sub>3D</sub> Avg. |
|--------------|-------------|----------------|--------------|--------------|--------------|--------------|
| InfraDet3D | South 1 | LiDAR | 75.81 | 47.66 | **42.16** | 55.21 |
| BEVFusionCoop | South 1 | LiDAR | 76.24 | 48.23 | 35.19 | 69.47 |
| CMTCoop | South 1 | LiDAR | **80.62** | **64.46** | **50.41** |**72.68** |
| InfraDet3D | South 2 | LiDAR | 38.92 | 46.60 | 43.86 | 43.13 |
| BEVFusionCoop | South 2 | LiDAR | 74.97 | 55.55 | 39.96 | 69.94 |
| CMTCoop | South 2 | LiDAR | **79.34** | **60.81** | **45.53** | **70.31** |
| InfraDet3D | South 1 | Camera + LiDAR | 67.08 | 31.38 | 35.17 | 44.55 |
| BEVFusionCoop | South 1 | Camera + LiDAR | 75.68 | 45.63 | **45.63** | 66.75 |
| CMTCoop | South 1 | Cam+LiDAR | **80.86** | **61.37** | 45.32 | **70.65** |
| InfraDet3D | South 2 | Camera + LiDAR | 58.38 | 19.73 | 33.08 | 37.06 |
| BEVFusionCoop | South 2 | Camera + LiDAR | 74.73 | **53.46** | **41.96** | 66.89 |
| CMTCoop | South 2 | Cam+LiDAR | **78.92** | 52.67 | 39.76 | **67.21** |
<!-- 75.81 47.66 42.16 55.21
BEVFusionCoop south 1 LiDAR 76.24 48.23 35.19 69.47
CMTCoop south 1 LiDAR 80.62 64.46 50.41 72.68
InfraDet3D south 2 LiDAR 38.92 46.60 43.86 43.13
BEVFusionCoop south 2 LiDAR 74.97 55.55 39.96 69.94
CMTCoop south 2 LiDAR 79.34 60.81 45.53 70.31
InfraDet3D south 1 Cam+LiDAR 67.08 31.38 35.17 44.55
BEVFusionCoop south 1 Cam+LiDAR 75.68 45.63 45.63 66.75
CMTCoop south 1 Cam+LiDAR 80.86 61.37 45.32 70.65
InfraDet3D south 2 Cam+LiDAR 58.38 19.73 33.08 37.06
BEVFusionCoop south 2 Cam+LiDAR 74.73 53.46 41.96 66.89
CMTCoop south 2 Cam+LiDAR 78.92 52.67 39.76 67.21 -->

### Visualization

#### Performance of Vehicular only model (CMT) from infrastructure perspective (left) and vehicular perspective (right)

[<img src="./figs/1688625741_352108537_vehicle_camera_basler_16mm.jpg" width="50%">](https://drive.google.com/file/d/1Te-xNnNR9YuGwsTHXI6lfpuQpcwbg6c8/view?usp=drive_link "CMT video")


#### Performance of Cooperative model (CMTCoop - left) vs. Vehicular only model (CMT - right) from infrastructure perspective.

[<img src="./figs/1688625741_338268958_s110_camera_basler_south2_8mm.jpg" width="50%">](https://drive.google.com/file/d/137nkUwPNB_2ygdDNXaRg9L9WOv3WG7hH/view?usp=sharing "CMT video")

<!-- ## Main Results
Results on nuScenes **val set**. The default batch size is 2 on each GPU. The FPS are all evaluated with a single Tesla A100 GPU. (15e + 5e means the last 5 epochs should be trained without [GTsample](https://github.com/junjie18/CMT/blob/master/projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py#L48-L87))

| Config            |Modality| mAP      | NDS     | Schedule|Inference FPS|
|:--------:|:----------:|:---------:|:--------:|:--------:|:--------:|
| [vov_1600x640](./projects/configs/camera/cmt_camera_vov_1600x640_cbgs.py) |C| 40.6% | 46.0%  | 20e | 8.4 |
| [voxel0075](./projects/configs/lidar/cmt_lidar_voxel0075_cbgs.py) |L| 62.14% | 68.6%    | 15e+5e | 18.1 |  
| [voxel0100_r50_800x320](./projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py)  |C+L| 67.9%     | 70.8%    | 15e+5e | 14.2 |
| [voxel0075_vov_1600x640](./projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py)  |C+L| 70.3% | 72.9%    | 15e+5e | 6.4 |

Results on nuScenes **test set**. To reproduce our result, replace `ann_file=data_root + '/nuscenes_infos_train.pkl'` in [training config](./projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py) with `ann_file=[data_root + '/nuscenes_infos_train.pkl', data_root + '/nuscenes_infos_val.pkl']`:

| Config            |Modality| mAP      | NDS     | Schedule|Inference FPS|
|:--------:|:----------:|:---------:|:--------:|:--------:|:--------:|
| [vov_1600x640](./projects/configs/camera/cmt_camera_vov_1600x640_cbgs.py) |C| 42.9% | 48.1%  | 20e | 8.4 |
| [voxel0075](./projects/configs/lidar/cmt_lidar_voxel0075_cbgs.py) |L| 65.3% | 70.1%    | 15e+5e | 18.1 | 
| [voxel0075_vov_1600x640](./projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py)  |C+L| 72.0% | **74.1%**    | 15e+5e | 6.4 | -->

## Resource

Refer the following links for other resources related to this project:
- [Initial presentation]()
- [Final report](./docs/IDP_Report.pdf)
- [Final presentation](https://docs.google.com/presentation/d/1H1zfHr0BnH_xsGI7ZzYMYv3Z0eJl02HU1f5R5RPwzvU/edit?usp=sharing)

## Citation
Please consider citing the original work on CMT if you find this work helpful.

<!-- ## Citation
If you find CMT helpful in your research, please consider citing: 
```bibtex   
@article{yan2023cross,
  title={Cross Modal Transformer via Coordinates Encoding for 3D Object Dectection},
  author={Yan, Junjie and Liu, Yingfei and Sun, Jianjian and Jia, Fan and Li, Shuailin and Wang, Tiancai and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2301.01283},
  year={2023}
}
``` -->

<!-- ## Contact
If you have any questions, feel free to open an issue or contact us at yanjunjie@megvii.com, liuyingfei@megvii.com, sunjianjian@megvii.com or wangtiancai@megvii.com. -->
