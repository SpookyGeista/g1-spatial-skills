# G1 Spatial Skills

Mapping, mission control, and terrain skills for the G1 in Isaac Sim. The project adds simulated LiDAR/IMU observations, localization integration, RGB-LiDAR visualization, and immersive RGB-D rendering.


[Tool guide](docs/TOOLS.md): commands, inputs, outputs, and operating sequence.

## Initialize

The standalone sensor tests require Python 3.11:

```bash
git clone https://github.com/SpookyGeista/g1-spatial-skills.git
cd g1-spatial-skills
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python -m unittest discover -s tests -p 'test_rgb_lidar_preview.py'
```

This first check does not start Isaac Sim or ROS.

## Configure mission execution

Full missions use an existing Isaac Sim / SONIC setup and ROS 2 Humble with FAST-LIO. Set the installation path in `punktnull/retained.py`, then the simulation and XR Python paths in `demo.sh`.

The retained runtime also needs these directories inside the SONIC checkout:

```text
data/null_exploration_2026-09-17/
data/null_vc_demo_2026-09-16/
```

These runtime snapshots are not included in this export. They are required in addition to the public SONIC checkout. The ROS build script also expects the FAST-LIO source and the Livox interface package named in `ros2_ws/build_local.sh`.

Once those prerequisites are available, build the ROS workspace:

```bash
bash ros2_ws/build_local.sh
source ros2_ws/activate.sh
```

## First mission

After configuring the runtime:

```bash
bash demo.sh run --gui --rgb-lidar
```

The launcher prints a new `runs/lio_...` directory. From a second terminal:

```bash
RUN_DIR=runs/lio_YYYYMMDD_HHMMSS  # replace with the printed run directory
bash demo.sh control "$RUN_DIR" start
bash demo.sh control "$RUN_DIR" abort
```

For immersive rendering, install `requirements-immersive.txt` into the dedicated XR environment and use `bash demo.sh run --immersive`. The renderer is a prototype; physical-headset validation is pending.
