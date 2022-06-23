# Introduce
This is about utilities for yolov5 release 6.1 include  train, export, data processing, etc.  
해당 디렉토리는 yolov5 릴리즈 6.1 버전과 관련된 내용이 있습니다. 네트워크 학습 환경 자체 혹은 학습을 하기위해 데이터를 가공하는 도구 등을 다룹니다. 

# Yolov5 

## Environment
This is my basic environments.
```
Ubuntu 18.04 

cuda 11.1
cudnn 8.2.1
+ tensorrt 8.0.3.4 (maybe no use)
python 3.7.13 (by anaconda3)
```
Cause using cuda 11.1 version, Downlaod pytorch manually with other info.
```shell
$ pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Installation
Using `pip` for setting environments.

| Module          |  Require   |      MyVersion       |
|-----------------|:----------:|:--------------------:|
| matplotlib      | \>= 3.2.2  |        3.5.2         |
| numpy           | \>= 1.18.5 |        1.21.6        |
| opencv-python   | \>= 4.1.2  |       4.5.5.64       |
| pillow          | \>= 7.1.2  |        9.1.1         |
| PyYAML          | \>= 5.3.1  |         6.0          |
| requests        | \>= 2.23.0 |        2.28.0        |
| scipy           | \>= 1.4.1  |        1.7.3         |
| tqdm            | \>= 4.41.0 |        4.64.0        |
| tensorboard     | \>= 2.4.1  |        2.9.1         |
| wandb           |     -      |       0.12.19        |
| pandas          | \>= 1.1.4  |        1.3.5         |
| seaborn         | \>= 0.11.0 |        0.11.2        |
| onnx            | \>= 1.9.0  |        1.11.0        |
| onnx-simplifier | \>= 0.3.6  |        0.3.10        |
| albumentations  | \>= 1.0.3  |        1.2.0         |
| cython          |     -      |       0.29.30        |
| pycocotools     |  \>= 2.0   |        2.0.4         |
| thop            |     -      | 0.1.0.post2206102148 |

# TF Record to Yolov5 Dataset
