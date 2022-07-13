# ROS2 PyTorch node

Run a PyTorch model in C++ in ROS2. A template node to be able to adapt to whatever model you want to run, for instance object detection of people and bikes. This specific implementation is focused on running the models with CUDA devices available in a docker environment.

To make life easier, I've created a docker environment that houses all the dependcies needed to run this package. For a desktop environment that decision was a no-brainer, and I would recommend that most folks go for that option. The only catch is that the current docker image is a hefty 16 Gigabytes in size due to it containing some pretty big packages and software such as ROS2 and libtorch in additon to the fact that the base cuda-ubuntu image it's built from is already ~5 Gigabytes.

If you do not have a CUDA supported device then you should still be able to port the package to your own environment, be sure to set the `GPU` parameter to 0 in the launch file!

## Prerequisites

This package was developed on a linux desktop with the following specifications:

- Ubuntu 20.04
- 12th Gen Intel(R) Core(TM) i7-12700K
- NVIDIA GeForce RTX 3060
- NVIDIA Driver Version 510.73.05

To build the docker image you will need:

1. A Linux Machine (GNU/Linux x86_64 with kernel version > 3.10).
2. NVIDIA GPU with Architecture >= Kepler (or compute capability 3.0)
3. NVIDIA Linux drivers >= 418.81.07
4. Docker >= 19.03 & Nvidia Container Toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


## Build and installation

### Docker 

*_NOTE:_* This build is centered around CUDA 11.3, since that version currently supports the RTX 3060 GPU architecture on my machine, this directly correlates to the CUDNN and pytorch and libtorch versions that are installed in the container.

1. Make all the shell scripts in the docker directory executable via `chmod +x build_container.sh entrypoint.sh run_container.sh`. You only need to do this once when you initially clone the repository.
2. Build the container using `./build_container.sh`, this will take some time and will need 16 Gigabytes.
3. Run the container using `./run_container.sh`

How you operate in the container is up to you. I use vscode since it has some pretty cool extensions that make development easy.

### ros2_pytorch node

Once you are in the container development environment:

Add Libtorch to your `CMAKE_PREFIX_PATH` env variable via the command below or preferably set the correct path in the `CMakeLists.txt` of the package, you will need to do this if you have change the Libtorch
installation location in the Dockerfile, otherwise it has already been set correctly:
```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/libtorch
```
Navigate to the root of the ROS2 workspace, this is created in the docker container:
```
cd /home/ros2_ws
```
Build the ROS2 environment, make sure add the option for the symbolic link install:
```
colcon build --symlink-install
```
Source the setup.bash file in the workspace:
```
source install/setup.bash
```
Launch the ros2_pytorch node:
```
ros2 launch ros2_pytorch ros2_pytorch.launch.py
```


