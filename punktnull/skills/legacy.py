"""Read-only access to retained controller/data; never import Isaac at module load."""
from pathlib import Path
import importlib.util
import os
import sys

OLD = Path(os.environ.get('PUNKTNULL_RETAINED_REPO', '/home/loya2/WholeBodyTele/GR00T-WholeBodyControl')).resolve()
WORKSPACE = Path(__file__).resolve().parents[2]
CRAWL = OLD / 'data/isaac_sonic_bones_crawl_2026-09-16'
EXPLORATION = OLD / 'data/null_exploration_2026-09-17'
LEGACY = OLD / 'data/null_vc_demo_2026-09-16'


def load(name, path):
    key = '_punktnull_retained_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def ground_module():
    # The retained guard imports only the exact SDK/IsaacLab joint permutation.
    if 'route_data' not in sys.modules:
        sys.modules['route_data'] = load('crawl_route_data', CRAWL / 'route_data.py')
    return load('crawl_ground_guard', CRAWL / 'ground_guard.py')


def planner_module():
    return load('planner_client', EXPLORATION / 'planner_client.py')
