## Unsupervised Domain Adaptation for Referring Semantic Segmentation

This is the official implementation of the CVMN paper:


## Installation
First, clone the repo locally:
```
git clone https://github.com/FenriartS/CVMN
```
Then, install PyTorch 1.8 and torchvision 0.9:
```
conda install pytorch==1.8.0 torchvision==0.9.0
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
If you encounter the problem of missing ytvos.py file, you can manually download the file from [here](https://github.com/youtubevos/cocoapi/tree/master/PythonAPI/pycocotools) and put it in the installed pycocotools folder.

Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

## Preparation
Download and extract 2021 version of Refer-Youtube-VOS train images from [RVOS](https://youtube-vos.org/dataset/rvos/). 
Follow the instructions [here](https://kgavrilyuk.github.io/publication/actor_action/) to download A2D-Sentences dataset.


## Training

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet101/50 --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path --output_dir /path/to/output_dir
```

## Inference

```
python inference.py --model_path /path/to/model_weights
```



