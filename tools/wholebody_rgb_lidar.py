#!/usr/bin/env python3
"""Launch the retained wholebody sender with an opt-in RGB-LiDAR display."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path('/home/loya2/WholeBodyTele/GR00T-WholeBodyControl')
ARCHIVE = REPOSITORY/'data/teleop_archives/g1_pose_2026-09-09'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Wholebody Isaac mission output directory')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read-only compatibility check; no XR or DDS started')
    mode.add_argument('--diagnose', action='store_true', help='Start real CloudXR for 3 seconds after initialization; no DDS sender')
    mode.add_argument('--display-only', action='store_true', help='Show VR display and tracking without starting the SONIC sender')
    parser.add_argument('--ipd-mm', type=float, default=64., help='Virtual eye separation, 50..80 mm; set to headset IPD')
    parser.add_argument('--max-age', type=float, default=2., help='Hide point cloud after this many wall-clock seconds')
    parser.add_argument('--immersive', action='store_true', help='Real OpenXR eye projections of dense metric RGB-D surfaces')
    parser.add_argument('--replay', action='store_true', help='Historical RGB-D for --immersive --diagnose only; no control')
    args = parser.parse_args()
    if not 50 <= args.ipd_mm <= 80 or not .1 <= args.max_age <= 5:
        parser.error('IPD must be 50..80 mm and max age .1..5 seconds')
    packet = args.run.resolve()/'sim/rgb_lidar/latest.npz'
    if args.replay and not (args.immersive and args.diagnose):
        parser.error('--replay requires --immersive --diagnose')
    if args.immersive:
        if args.ipd_mm != 64.:
            parser.error('Im immersiven Modus kommen Augenabstand und Augenposen aus OpenXR; --ipd-mm weglassen.')
        packet = args.run.resolve()/'sim/rgbd/latest.npz'
        sys.path.insert(0, str(ROOT/'.runtime/immersive'))
        try:
            import xr
            import vulkan
        except ImportError:
            parser.error('Immersive dependency missing. Install requirements-immersive.txt into .runtime/immersive (see docs).')
    if not args.run.is_dir():
        parser.error('Laufordner existiert nicht: '+str(args.run))
    client_path = ARCHIVE/'gear_sonic/utils/teleop/isaac_teleop_client.py'
    if not client_path.is_file():
        parser.error('Retained wholebody sender is missing')
    sys.path[:0] = [str(ROOT), str(ARCHIVE), str(ARCHIVE/'external_dependencies/unitree_sdk2_python')]
    import isaacteleop.viz as viz
    import isaacteleop.oxr as oxr
    import gear_sonic.utils.teleop.isaac_teleop_client as client
    from punktnull.wholebody_lidar_xr import client_with_display
    for name in ('get_oxr_handles', 'begin_frame', 'end_frame', 'destroy'):
        if not hasattr(viz.VizSession, name):
            raise RuntimeError('Installed Televiz lacks '+name)
    patched = client_with_display(client.IsaacTeleopClient, packet,
        ipd_m=args.ipd_mm/1000., max_age_s=args.max_age, immersive=args.immersive, replay=args.replay)
    if args.check:
        from importlib.metadata import version
        print(json.dumps(dict(status='compatible', isaacteleop=version('isaacteleop'),
            source=str(client_path), source_sha256=hashlib.sha256(client_path.read_bytes()).hexdigest(),
            packet=str(packet), packet_exists=packet.exists(), session_handles=hasattr(oxr, 'OpenXRSessionHandles'),
            scope='Import/API check only; no GPU, XR, DDS, tracking or headset test'), indent=2))
        return
    if not args.diagnose:
        from punktnull.rgb_lidar import read_packet
        from punktnull.rgbd_surface import read_rgbd
        import zipfile
        try:
            (read_rgbd if args.immersive else read_packet)(packet, max_age_s=args.max_age)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            parser.error('Keine frischen Daten. Simulation mit '+('--immersive' if args.immersive else '--rgb-lidar')+' starten, READY abwarten und den ausgegebenen Laufordner verwenden. '+str(exc))
    print(('Immersive RGB-D-Quelle: ' if args.immersive else 'RGB-LiDAR-Quelle: ')+str(packet), flush=True)
    if args.diagnose or args.display_only:
        display = patched()
        try:
            display.start_streaming()
            deadline = time.monotonic()+3. if args.diagnose else float('inf')
            while time.monotonic() < deadline:
                time.sleep(.1)
                if display._display_error:
                    raise RuntimeError(display._display_error)
                if not display._display_thread.is_alive():
                    raise RuntimeError('XR-Sitzung wurde beendet')
        except KeyboardInterrupt:
            pass
        finally:
            display.close()
        if args.diagnose:
            if display._display_thread.is_alive():
                raise RuntimeError('XR-Thread beim Beenden nicht gestoppt')
            if args.immersive:
                stats = getattr(display, '_immersive_stats', {})
                print(json.dumps(stats, indent=2), flush=True)
                if not stats.get('projection_frames'):
                    raise RuntimeError('No stereo projection frames submitted; headset/runtime may not be ready')
                if args.replay and not stats.get('triangles'):
                    raise RuntimeError('No reconstructed RGB-D surfaces rendered')
            print('CloudXR-Start, Frame-Schleife und Beenden bestanden. Bild in der Brille NICHT geprüft.', flush=True)
        return
    client.IsaacTeleopClient = patched
    os.chdir(ARCHIVE)
    # Same options as the retained start-3-meta-quest.sh, with one display adapter.
    sys.argv = ['pico_manager_thread_server', '--manager', '--input-source', 'isaac-teleop', '--vis_smpl', '--vis_vr3pt']
    runpy.run_module('gear_sonic.scripts.pico_manager_thread_server', run_name='__main__')


if __name__ == '__main__':
    main()
