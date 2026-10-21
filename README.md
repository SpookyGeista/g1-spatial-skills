# G1 Spatial Skills

Mapping, mission control, and terrain skills for the G1 in Isaac Sim.

- Simulated LiDAR/IMU data and localization integration.
- Terrain-skill execution and validation.
- RGB-LiDAR visualization and immersive RGB-D rendering.

## Usage

Application modules are in `punktnull/`; validation tools are in `tools/`.
`demo.sh` provides simulation and visualization commands.

```bash
python -m unittest discover -s tests -p 'test_rgb_lidar_preview.py'
```

The immersive renderer is a prototype; physical-headset validation is pending.
