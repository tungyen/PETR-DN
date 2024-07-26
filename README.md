# _Introduction_ #
In this repository, the project folder is PETR, the other files are implementation of PETR for Kitti Dataset, which lead to bad result. The update for double camera case will be done in the future I hope. 

Before running the project, remember to download NuScenes Dataset following the official git repository of PETR first. The report of this project is at DN_PETR.pdf

![image](https://github.com/tungyen/PETR-DN/blob/tungyen/stat.png)
![image](https://github.com/tungyen/PETR-DN/blob/tungyen/pic.png)

# _PETR package and install_ #

## _To run the project, cd into PETR folder first_ ##
```bash
cd PETR
```

## _install basic package_ ##
```bash
pip install ninja wheel lyft_dataset_sdk networkx==2.2 numba==0.48.0 nuscenes-devkit plyfile scikit-image tensorboard trimesh==2.35.39 numpy==1.19.5 open3d einops yapf==0.40.0
```

## _install mmcv, mmdet, mmseg_ ##
```bash
pip install --upgrade pip
```
```bash
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
```

## _If encountering public key problem when installing git_ ##
```bash
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
```

## _When encountering installing package with timezone_ ##
```bash
ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
DEBIAN_FRONTEND=noninteractive apt-get install gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget -y
```

## _Install github_ ##
```bash
apt-get update && apt-get install -y git
```

## _install mmdet3d_ ##
```bash
cd mmdetection3d
git checkout v0.17.1 
pip install -r requirements/build.txt
python3 setup.py develop
```

## _For uncompressing .tgz file_ ##
```bash
tar -xvzf /path/to/yourfile.tgz
```

## _Downgrade the setpool before running any code_ ##
```bash
pip install setuptools==59.5.0
```

## _If encountering some module import error when running python code for preparing data_ ##
```bash
PYTHONPATH=${PWD}:$PYTHONPATH python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

# _Running the trainning code_ #

## _VovNet 800*320 for PETR_ ##
```bash
tools/dist_train.sh projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py 1 --work-dir output/epoch24/petr_vovnet_gridmask_p4_800x320/
```

## _VovNet 800*320 for DN-PETR_ ##
```bash
tools/dist_train.sh projects/configs/denoise/petr_vovnet_gridmask_p4_800x320_dn.py 1 --work-dir output/epoch24/petr_vovnet_gridmask_p4_800x320_dn/
```

