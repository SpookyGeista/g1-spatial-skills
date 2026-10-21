#!/usr/bin/env bash
# Source into a shell; all paths are local to this project. No global setup changes.
NULL_LIO_WS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /opt/ros/humble/setup.bash
export CMAKE_PREFIX_PATH="$NULL_LIO_WS/install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export AMENT_PREFIX_PATH="$NULL_LIO_WS/install${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$NULL_LIO_WS/install/local/lib/python3.10/dist-packages:$NULL_LIO_WS/install/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$NULL_LIO_WS/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_LOCALHOST_ONLY=1
