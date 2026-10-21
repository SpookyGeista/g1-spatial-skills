#!/usr/bin/env bash
set -eo pipefail
WS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=$(dirname -- "$WS")
source /opt/ros/humble/setup.bash
export CMAKE_PREFIX_PATH="$WS/install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export AMENT_PREFIX_PATH="$WS/install${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="$WS/install/local/lib/python3.10/dist-packages:$WS/install/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$WS/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$WS/logs" "$WS/runtime/Log" "$WS/runtime/PCD"
/usr/bin/cmake -S "$WS/src/livox_ros_driver2_interfaces_only" -B "$WS/build/livox_interfaces" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$WS/install" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
/usr/bin/cmake --build "$WS/build/livox_interfaces" --parallel 2
/usr/bin/cmake --install "$WS/build/livox_interfaces"
/usr/bin/cmake -S "$PROJECT/third_party/FAST_LIO_ROS2" -B "$WS/build/fast_lio" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$WS/install" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
  -DFASTLIO_OUTPUT_ROOT="$WS/runtime" -DBUILD_TESTING=OFF
/usr/bin/cmake --build "$WS/build/fast_lio" --parallel 2
/usr/bin/cmake --install "$WS/build/fast_lio"
