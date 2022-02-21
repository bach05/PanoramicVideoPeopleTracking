# Panoramic Video People Tracking 
This work builds on top of [Monocular Person Tracking and Identification framework](https://doi.org/10.1016/j.robot.2019.103348), proposed by Koide at. al, to adapt it working with panoramic images. The main advantage of using panoramic video is that people cannot leave the field of view of the camera (if not occluded). This fact leads to a better tracking stability. 

<p align="center">
  <img src="https://github.com/bach05/PanoramicVideoPeopleTracking/blob/master/media/PanoramicPeopleTracking_example.gif?raw=true" />
</p>
<p align="center">
<em>Example of our Panoramic People Tracking</em>
</p>

We also provide **a dataset of panoramic videos**, labelled everi 10-20 frames with a target subject. The dataset is available [here for download](https://mega.nz/file/V2wi0RgD#YwGp4fXdiwiH7WKC2riCzbe4NftLOx2zKZowFW8GVAM). The password is `iaslabPVPT`.


If you use our algorithms or our dataset in your reasearch, please cite our article: 

```
under review [...] 
```
The work was develepoed in collaboration with [filber97](https://github.com/filber97).

# Installation

**ATTENTION:** we are working on some installation and compatibility issue. At the moment, we tested it on ROS Melodic. 

#### Update system

```bash
sudo apt-get update
sudo apt-get upgrade
```

#### Flann & OpenBlas

```bash
sudo apt-get install libflann-dev libopenblas-base libopenblas-dev
```

#### dlib

```bash
mkdir ~/libs && cd ~/libs

wget http://dlib.net/files/dlib-19.17.tar.bz2
tar xvf dlib-19.17.tar.bz2

echo "export DLIB_ROOT=~/dlib-19.17" >> ~/.bashrc
source ~/.bashrc
```

#### tensorflow

```bash
sudo apt install python-pip
sudo pip install -U pip setuptools numpy
sudo pip install -U --extra-index-url https://developer.download.nvidia.com/compute/redist/jp33 tensorflow-gpu
```

#### ROS

See http://wiki.ros.org/melodic/Installation/Ubuntu

!! Install tensorflow before ROS otherwise you see some dependency issues...

#### tf-pose-estimation

```bash
# llvmlite requires llvm >= 7.0
# for Xavier
sudo apt install llvm-7.0

# for TX2 with Jetpack 3.3
mkdir ~/libs && cd ~/libs
wget http://releases.llvm.org/7.0.1/llvm-7.0.1.src.tar.xz
tar xvf tar xvf llvm-7.0.1.src.tar.xz
cd llvm-7.0.1.src && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_LINKER=gold -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j6
sudo make install
```

```bash
cd catkin_ws/src
git clone https://github.com/koide3/tf-pose-estimation

# Follow the official installation instruction
cd tf-pose-estimation
sudo pip install -r requirements.txt

# In case you meet scipy install issue
# sudo apt-get install gfortran

cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python setup.py build_ext --inplace

cd ../..
sudo python setup.py install

# You may need to comment out python package dependencies in setup.py like as follows:
# 21: #   'dill==0.2.7.1',

cd models/graph/cmu
bash download.sh
```

#### other packages

```bash
cd catkin_ws/src
git clone https://github.com/koide3/open_face_recognition.git
git clone https://github.com/koide3/ccf_person_identification.git

sudo apt-get install ros-kinetic-image-transport ros-kinetic-image-transport-plugins ros-melodic-image-proc
```
#### Principal packages

```bash
git clone https://github.com/bach05/PanoramicVideoPeopleTracking.git 
catkin build
```

# Usage

## Detection with ROI
```bash
roslaunch tracking_demo omnidirectional_roi.launch
roslaunch tracking_demo start_robot.launch webcam:=true publish_dummy_frames:=true
```

## Detection with TILES
```bash
roslaunch tracking_demo omnidirectional_tiles.launch
roslaunch tracking_demo start_robot.launch webcam:=true publish_dummy_frames:=true
```