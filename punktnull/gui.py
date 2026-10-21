"""Retain the existing map widget, correct its old ideal-pose attribution.

Only two text literals in the constructor change. No camera, render, physics,
map pixels, operator callbacks or mission decisions are changed.
"""
import hashlib
import inspect
import textwrap
from pathlib import Path
from .retained import prepare
prepare()
import gui_view as retained

_original=textwrap.dedent(inspect.getsource(retained.ObservedMapPanel.__init__))
_source=_original
_replacements=[
    ('Ideal measured simulator pose, not SLAM','FAST-LIO estimated pose; one-time initial registration; RGB-D occupancy'),
    ('RGB-D / ideal simulator localization','RGB-D / FAST-LIO localization'),
]
for before,after in _replacements:
    if _source.count(before)!=1:raise RuntimeError('Retained GUI attribution changed; review override')
    _source=_source.replace(before,after)
_namespace=dict(retained.__dict__)
exec(compile(_source,retained.__file__+':lio_attribution','exec'),_namespace)
_constructor=_namespace['__init__']


class ObservedMapPanel(retained.ObservedMapPanel):
    def __init__(self,run,output):
        _constructor(self,run,output)
        self._manifest['display_attribution_patch']={
            'original_constructor_sha256':hashlib.sha256(_original.encode()).hexdigest(),
            'replacement_constructor_sha256':hashlib.sha256(_source.encode()).hexdigest(),
            'exact_text_replacements':_replacements,
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'map_pixels_or_physics_changed':False,
        }
        self._save_manifest()
