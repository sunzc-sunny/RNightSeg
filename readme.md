# Nighttime Image Semantic Segmentation with Retinex Theory
Paper: https://www.sciencedirect.com/science/article/pii/S0262885624002543


## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model](#model)
- [Demo](#demo)

## Installation

Please follow the guide to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/sunzc-sunny/RNightSeg.git

# Follow the mmseg0.x to create environment
https://github.com/open-mmlab/mmsegmentation/blob/0.x/docs/en/get_started.md#installation
```


## Dataset

**NightCity+**  <br>
Please download the offical NightCity+ benchmark from NightLab [GitHub](https://github.com/xdeng7/NightLab).

**BDD100K-Night** <br>
Please download the offical BDD100K-Night benchmark from NightCity [Homepage](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html).

## Usage
Update the dataset path in the config files<br>
Training the methods RNightSeg on NightCity+ datasets.
```bash
python tools/train.py configs/RNightSeg_segformer.py
or 
bash tools/dist_train.sh configs/RNightSeg_segformer.py <num_gpus>
```
Inference the method RNightSeg on NightCity+ datasets.
```bash
python tools/test.py configs/RNightSeg_segformer.py workdirs/RNightSeg_segformer/latest.pth --eval mIoU
or
bash tools/dist_test.sh configs/RNightSeg_segformer.py workdirs/RNightSeg_segformer/latest.pth <num_gpus> --eval mIoU
```

## Training log
[RNightSeg_SegFormer](./logs/RNightSeg_deeplabv3plus.log)
<br>
[RNightSeg_DeepLabV3+](./logs/RNightSeg_segformer.log)


## Demo

[Real video segmentation results](./videos/seg_video_1.mp4)
