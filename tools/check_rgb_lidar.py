#!/usr/bin/env python3
"""Inspect a live or explicitly replayed cloud; optionally exercise Televiz GPU."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from punktnull.rgb_lidar import read_packet
from punktnull.lidar_stereo import StereoWindow
from punktnull.wholebody_lidar_xr import device_buffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true', help='Inspect historical capture without republishing it as live')
    parser.add_argument('--gpu', action='store_true', help='Render with installed Televiz offscreen; no XR session')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    xyz, rgb, metadata = read_packet(args.packet, replay=args.replay)
    args.output.mkdir(parents=True)
    viewer = StereoWindow()
    start = time.monotonic()
    left, right = viewer.render(xyz, rgb, [0, 0, 0], [1, 0, 0, 0])
    elapsed = time.monotonic()-start
    Image.fromarray(np.concatenate((left, right), axis=1)).save(args.output/'stereo.png')
    gpu_pixels = None
    if args.gpu:
        import isaacteleop.viz as viz
        cfg = viz.VizSessionConfig()
        cfg.mode = viz.DisplayMode.kOffscreen
        cfg.window_width, cfg.window_height = viewer.width, viewer.height
        session = viz.VizSession.create(cfg)
        try:
            config = viz.QuadLayerConfig()
            config.name = 'RGB-LiDAR diagnostic'
            config.resolution = viz.Resolution(viewer.width, viewer.height)
            config.stereo = True
            config.generate_mipmaps = False
            layer = session.add_quad_layer(config)
            # Exercise placement as used by the XR path, then use fullscreen
            # for deterministic readback. set_visible is broken in this SDK.
            layer.set_placement(viz.QuadLayerPlacement(viz.Pose3D((0, 0, -1), (1, 0, 0, 0)), viewer.size))
            layer.set_placement(None)
            left_buffer, left_owner = device_buffer(viz, left)
            right_buffer, right_owner = device_buffer(viz, right)
            layer.submit(left_buffer, right_buffer)
            session.render()
            gpu_pixels = np.asarray(session.readback_to_host()).copy()
            Image.fromarray(gpu_pixels).save(args.output/'televiz.png')
            blank_buffer, blank_owner = device_buffer(viz, np.zeros_like(left))
            layer.submit(blank_buffer, blank_buffer)
            session.render()
            cleared = np.asarray(session.readback_to_host()).copy()
            if np.array_equal(cleared, gpu_pixels):
                raise RuntimeError('Clearing stale stereo image did not change GPU output')
        finally:
            session.destroy()
    report = dict(source=str(args.packet.resolve()), replay=args.replay, metadata=metadata,
        points=len(xyz), stereo_render_ms=elapsed*1000., eye_images_differ=not np.array_equal(left, right),
        visible_pixels=int(np.any(left[..., :3] != [8, 13, 20], axis=2).sum()),
        gpu_offscreen_executed=gpu_pixels is not None, headset_tested=False,
        scope='Spatial stereo window; virtual IPD 64mm; not a 360-degree projection layer')
    if gpu_pixels is not None:
        # Same-size fullscreen offscreen draw should preserve coloured pixels.
        report['gpu_nonuniform'] = bool(gpu_pixels[..., :3].std() > 1.)
        if not report['gpu_nonuniform']:
            raise RuntimeError('GPU readback contains no useful image')
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['points'] or not report['visible_pixels']:
        raise RuntimeError('No visible coloured LiDAR points')


if __name__ == '__main__':
    main()
