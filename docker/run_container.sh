#!/bin/bash

docker run \
    -it --rm \
    --name="ros2_torch_container" \
    --volume="$PWD/../ros2_pytorch:/home/ros2_ws/src/ros2_pytorch/:rw" \
    --runtime=nvidia \
    ros2_torch:latest
