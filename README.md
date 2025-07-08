# When Anchors Meet Cold Diffusion: A Multi-Stage Approach to Lane Detection

## Installation

### Create a conda virtual environment and activate it
```Shell
conda env create -f CDiffLane.yml
conda activate CDiffLane
```
### Data preparation

#### CULane
Create `dataset` directory.

Download [CULane](https://xingangpan.github.io/projects/CULane.html) and place in `dataset` directory.


For CULane, you should have structure like this:
```
dataset/culane/driver_xx_xxframe    # data folders x6
dataset/culane/laneseg_label_w16    # lane segmentation labels
dataset/culane/list                 # data lists
```


#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) alse place in `dataset` directory.

For Tusimple, you should have structure like this:
```
dataset/tusimple/clips # data folders
dataset/tusimple/lable_data_xxxx.json # label json file x4
dataset/tusimple/test_tasks_0627.json # test tasks json file
dataset/tusimple/test_label.json # test label json file
```

## Model preparation
Create `pretrain` directory.

We provide our pretrain model at [CDiffLane](https://drive.google.com/file/d/1aqFi3v2qL5G-gxS1VCyHkJU0zDGuq5Ad/view?usp=sharing), download and place in `pretrain` directory

## Getting Started
### Training
``` sh
python tools/train.py configs/CDiffLane/culane/CDiffLane_culane_dla34.py
```
### Validation
``` sh
python tools/test.py configs/CDiffLane/culane/CDiffLane_culane_dla34.py pretrain/cd_lane.pth
```
