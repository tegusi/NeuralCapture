# Neural Total Capture of 3D Human from Monocular Video

This repository provides a reference implementation for the ECCV 2022 paper *Neural Capture of Animatable 3D Human from Monocular Video*.


![](media/teaser.gif)


## Installation

There are two ways to set up the environment.

### *Docker installation* 

First you need to [install](https://docs.docker.com/engine/install/) docker.

```
# Pull the docker image
docker pull eximple/neural_total_capture:base
# Run the docker container
docker run --name neural_total_capture -it --mount type=bind,source=/${pwd},target=/workspace --gpus all -d eximple/neural_total_capture:base
# Start the interactive bash
docker exec -it neural_total_capture /bin/bash
```

Then you can do everything you want under `/workspace` directory.

### *Local installation*

You can also install the required packages locally. Our experiment platform is `Ubuntu 18.04` and `cuda 11.4`, and the  conda environment file is located in `docker/environment.yml`. You can build the environment by:

```
conda env create -f docker/env.yaml
conda activate neural_total_capture
```

## Data

We organize the data as the following structure, and you can prepare your own dataset like this. **Please note follow the same coordinate convention of your data, and I strongly recommend to visualize 2D keypoints at first**:

```                    
├── data
│   ├── snapshot_female_1  # dataset name 
│   │   ├── mask    # silhouette
│   │   │   └── mask_frame_1.png
│   │   ├── obj     # ground truth mesh vertices (optional)
│   │   │   └── smpl_frame_1.obj
│   │   ├── param   # SMPL parameters
│   │   │   └── smpl_frame_1.npz
│   │   ├── rgb     # image
│   │   │   └── rgb_frame_1.png
│   │   └── camera.mat  # camera parameters
│   └── smpl  # smpl related files
```

We provide the processed People-Snapshot dataset, smpl parameter and pretrained checkpoint, [here](https://drive.google.com/drive/folders/1UHKTtxznHH5iTSmhmE_mC2tnVJudbrjv?usp=sharing) is the link. Please move the dataset files to `data/` and checkpoint files to `logs/`.

## Train & Test

Train with single gpu:
```
python train.py --config=./example.ini 
```

You can also pass extra arguments following the config file such as, **Please note that the prepend [DEFAULT] in the configure file is a must**
```
python train.py --config=./example.ini --train_rays=4096
```

Train with multiple gpu:
```
python -m torch.distributed.launch --nproc_per_node=4 train_dist.py --datadir ./data/female-3-casual --output_dir ./logs
```

Test the model with checkpoint
```
python train.py --datadir ./data/female-3-casual --checkpoint snapshot_female_3.tar
```
