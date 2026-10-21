#!/usr/bin/env python3
"""Render real RGB-D surfaces on the GPU; explicit replay, no XR or control."""
import argparse
import json
from pathlib import Path
import sys

import glfw
import numpy as np
from OpenGL import GL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from punktnull.rgbd_surface import read_rgbd, write_rgbd, surface_mesh, pose_matrix, projection
from punktnull.immersive_gl import SurfaceRenderer, EyeFramebuffer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--packet', type=Path)
    source.add_argument('--recorded-sensor', type=Path)
    p.add_argument('--replay', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output directory')
    if args.recorded_sensor and not args.replay:
        p.error('--recorded-sensor requires --replay')
    args.output.mkdir(parents=True)
    packet = args.packet
    if args.recorded_sensor:
        with np.load(args.recorded_sensor, allow_pickle=False) as data:
            rgb, depth, meta = data['rgb'], data['depth'], json.loads(str(data['metadata']))
            packet = args.output/'rgbd.npz'
            write_rgbd(packet, rgb, depth, meta)
    rgb, depth, meta = read_rgbd(packet, replay=args.replay)
    vertices, indices = surface_mesh(rgb, depth, meta['intrinsics'])
    if not len(indices):
        raise RuntimeError('No observed surfaces')
    if not glfw.init():
        raise RuntimeError('GLFW initialization failed')
    glfw.window_hint(glfw.VISIBLE, False)
    glfw.window_hint(glfw.DOUBLEBUFFER, False)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    width, height = 640, 480
    window = glfw.create_window(width, height, 'Immersive surface verification', None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError('OpenGL window creation failed')
    try:
        glfw.make_context_current(window)
        renderer = SurfaceRenderer()
        framebuffer = EyeFramebuffer(width, height)
        try:
            renderer.upload(vertices, indices)
            framebuffer.bind()
            projection_matrix = projection(np.radians([-45, 45, 36.87, -36.87]))
            frames = []
            for eye_x in (-.032, .032, .20):
                camera = pose_matrix([eye_x, 0, 0], [1, 0, 0, 0])
                renderer.draw(projection_matrix @ np.linalg.inv(camera))
                GL.glFinish()
                frames.append(framebuffer.pixels()[..., :3])
            Image.fromarray(frames[0]).save(args.output/'front.png')
            Image.fromarray(np.concatenate(frames[:2], axis=1)).save(args.output/'stereo.png')
            Image.fromarray(frames[2]).save(args.output/'translated.png')
            report = dict(source=str(packet.resolve()), replay=args.replay, triangles=len(indices)//3,
                stereo_differs=bool(np.any(frames[0] != frames[1])),
                translation_changes_view=bool(np.any(frames[0] != frames[2])),
                visible_pixels=int(np.any(frames[0] != frames[0][0, 0], axis=-1).sum()),
                renderer=GL.glGetString(GL.GL_RENDERER).decode(), headset_tested=False,
                geometry='RGB-D measured front surface, metres; no occlusion filling', metadata=meta)
            if not report['stereo_differs'] or not report['translation_changes_view'] or not report['visible_pixels']:
                raise RuntimeError('GPU stereo/parallax verification failed')
            (args.output/'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        finally:
            framebuffer.close()
            renderer.close()
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == '__main__':
    main()
