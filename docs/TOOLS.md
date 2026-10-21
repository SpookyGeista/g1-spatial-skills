# Mission and visualization tools

Configure the retained runtime and ROS workspace described in the [README](../README.md) before full missions. Run the following commands from the project root.

## Start a mission and control it

```bash
bash demo.sh run --gui --rgb-lidar
```

The launcher prints a unique run directory. In a second terminal, replace the example directory below with that path:

```bash
RUN_DIR=runs/lio_YYYYMMDD_HHMMSS
bash demo.sh control "$RUN_DIR" start
bash demo.sh control "$RUN_DIR" takeover
bash demo.sh control "$RUN_DIR" release
bash demo.sh control "$RUN_DIR" abort
```

`start` begins the mission, `takeover` requests operator control, `release` returns control, and `abort` requests termination. These are explicit state requests for that one simulation run.

## Read-only PC preview

```bash
python tools/view_rgb_lidar.py --run "$RUN_DIR" --port 8766
```

Open the local preview address printed by the server. Add `--replay` when viewing retained data. This viewer does not issue robot commands.

## XR display

```bash
bash demo.sh xr --run "$RUN_DIR" --check
bash demo.sh xr --run "$RUN_DIR" --display-only
```

The check reports required runtime/configuration files. The second command starts the display path with the existing tracking/XR environment. For dense surfaces, start a mission with `bash demo.sh run --immersive`, then use:

```bash
bash demo.sh xr --run "$RUN_DIR" --immersive --display-only
```

Install `requirements-immersive.txt` in the dedicated renderer environment first. Physical-headset image and latency validation remain pending.

## Isolated platform trial

```bash
python tools/platform_trial.py --check
bash demo.sh climb --gui
```

This is a specific 12 cm platform fixture. It requires its retained scene/reference assets and is separate from general staircase behavior. It creates a separate `runs/platform_...` directory.

Sensor, scene, and timing probes under `tools/` use their own recorded fixtures. A fresh clone supports the README's small standalone sensor test; full mission commands need the additional runtime snapshots.
