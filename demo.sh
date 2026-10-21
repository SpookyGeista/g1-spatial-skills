#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mission_python=/home/loya2/WholeBodyTele/GR00T-WholeBodyControl/.venv_sim/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
case "${1:-help}" in
  run)
    shift
    mission_run_dir="$project_dir/runs/lio_$(date +%Y%m%d_%H%M%S)"
    printf 'Run directory: %s\n' "$mission_run_dir"
    exec "$mission_python" -B "$project_dir/run_mission.py" --output "$mission_run_dir" --record "$@"
    ;;
  control)
    shift
    exec "$mission_python" -B "$project_dir/tools/operator.py" "$@"
    ;;
  view)
    shift
    export PYTHONDONTWRITEBYTECODE=1
    exec /home/loya2/WholeBodyTele/GR00T-WholeBodyControl/.venv_teleop/bin/python "$project_dir/tools/view_rgb_lidar.py" "$@"
    ;;
  xr)
    shift
    export PYTHONDONTWRITEBYTECODE=1
    exec /home/loya2/WholeBodyTele/GR00T-WholeBodyControl/.venv_teleop/bin/python "$project_dir/tools/wholebody_rgb_lidar.py" "$@"
    ;;
  climb)
    shift
    platform_run_dir="$project_dir/runs/platform_$(date +%Y%m%d_%H%M%S)"
    printf 'Isolated 12-cm platform test. Run directory: %s\n' "$platform_run_dir"
    exec "$mission_python" -B "$project_dir/tools/platform_trial.py" --execute --record --output "$platform_run_dir" "$@"
    ;;
  *)
    echo 'Simulation: bash demo.sh run [--gui] [--auto-start] [--test steering]'
    echo 'Operator:   bash demo.sh control runs/<run> start|takeover|release|abort'
    echo 'RGB LiDAR:  bash demo.sh run --gui --rgb-lidar'
    echo 'Immersive:  bash demo.sh run --immersive'
    echo 'VR display: bash demo.sh xr --run runs/<run> --immersive --display-only'
    echo 'PC preview: bash demo.sh view --run runs/<run> [--replay]'
    echo 'Wholebody:  bash demo.sh xr --run runs/<run> [--check] [--ipd-mm 64]'
    echo 'Step-up:    bash demo.sh climb [--gui] [--start-y 0.02] [--start-yaw 1]'
    echo 'Step-up is an isolated 12-cm platform test, not general stairs; abort with touch runs/<run>/abort.request.'
    echo 'Reset: abort/close the current simulation, then launch a new run; evidence is retained.'
    ;;
esac
