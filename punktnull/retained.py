"""Explicit read-only reuse of the executed local Isaac / SONIC prototype."""
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/loya2/WholeBodyTele/GR00T-WholeBodyControl')
EXPLORATION=REPO/'data/null_exploration_2026-09-17'
LEGACY=REPO/'data/null_vc_demo_2026-09-16'
URDF=REPO/'gear_sonic/data/assets/robot_description/urdf/g1/main.urdf'

def prepare():
    for path in (EXPLORATION,LEGACY,REPO):
        if not path.is_dir():raise FileNotFoundError('Required retained project unavailable: '+str(path))
    for path in reversed((EXPLORATION,LEGACY,REPO)):
        if str(path) not in sys.path:sys.path.insert(0,str(path))

def load(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
