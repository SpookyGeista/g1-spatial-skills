"""Optional display adapter for the retained SONIC IsaacTeleopClient.

One Viz-owned OpenXR session is shared with the original DeviceIO trackers.
All XR calls belong to one thread; SONIC reads fresh snapshots without waiting
for display IO. No imports here start CloudXR, DDS, or a robot controller.
"""
from contextlib import ExitStack
import threading
import time
import zipfile

from .lidar_stereo import StereoWindow
from .rgb_lidar import read_packet


def device_buffer(viz, rgba):
    """Upload RGBA and return both the descriptor and its owning CUDA tensor."""
    import torch
    owner = torch.from_numpy(rgba).to(device='cuda', non_blocking=False).contiguous()
    buffer = viz.VizBuffer()
    buffer.data = int(owner.data_ptr())
    buffer.width, buffer.height = rgba.shape[1], rgba.shape[0]
    buffer.pitch = rgba.strides[0]
    buffer.format = viz.PixelFormat.kRGBA8
    buffer.space = viz.MemorySpace.kDevice
    return buffer, owner


def client_with_display(original, packet_path, *, ipd_m=.064, max_age_s=2., immersive=False, replay=False):
    """Return an opt-in subclass. The upstream client file remains untouched."""
    class LidarIsaacTeleopClient(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._display_stop = threading.Event()
            self._display_ready = threading.Event()
            self._display_lock = threading.Lock()
            self._display_thread = None
            self._display_latest = None
            self._display_stamp = 0.
            self._display_error = None

        def start_streaming(self):
            if self._display_thread is not None and self._display_thread.is_alive():
                return
            self._display_stop.clear()
            self._display_ready.clear()
            self._display_error = None
            self._display_thread = threading.Thread(target=self._display_loop, name='wholebody-lidar-xr', daemon=True)
            self._display_thread.start()
            if not self._display_ready.wait(45.):
                self.close()
                raise RuntimeError('CloudXR display initialization timed out; connect the headset')
            if self._display_error is not None:
                raise RuntimeError('CloudXR display initialization failed: '+self._display_error)

        def _display_loop(self):
            import numpy as np
            try:
                if immersive:
                    from .immersive_xr import run_immersive
                    run_immersive(self, packet_path, max_age_s, replay=replay)
                    return
                import isaacteleop.viz as viz
                import isaacteleop.deviceio as deviceio
                import isaacteleop.oxr as oxr
                import torch
                from isaacteleop.cloudxr import CloudXRLauncher
                # Initialize CUDA before publishing any tracking snapshots.
                print('RGB-LiDAR: GPU und CloudXR werden gestartet …', flush=True)
                torch.empty(1, device='cuda')
                with ExitStack() as stack:
                    launcher = CloudXRLauncher(install_dir=self._cloudxr_install_dir,
                        env_config=self._cloudxr_env_config, accept_eula=True,
                        setup_oob=self._use_adb, usb_local=self._use_adb)
                    stack.callback(launcher.stop)
                    head, hands = deviceio.HeadTracker(), deviceio.HandTracker()
                    controllers, body = deviceio.ControllerTracker(), deviceio.FullBodyTrackerPico()
                    trackers = [head, hands, controllers, body]
                    cfg = viz.VizSessionConfig()
                    cfg.app_name = self._app_name+'RGBLiDAR'
                    cfg.mode = viz.DisplayMode.kXr
                    cfg.xr_system_wait_seconds = 30
                    cfg.required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
                    print('RGB-LiDAR: OpenXR-Sitzung wird geöffnet …', flush=True)
                    session = viz.VizSession.create(cfg)
                    stack.callback(session.destroy)
                    handles = oxr.OpenXRSessionHandles(*session.get_oxr_handles())
                    tracking = stack.enter_context(deviceio.DeviceIOSession.run(trackers, handles))
                    window = StereoWindow(ipd_m=ipd_m)
                    layer_cfg = viz.QuadLayerConfig()
                    layer_cfg.name = 'G1 RGB LiDAR — simulation'
                    layer_cfg.resolution = viz.Resolution(window.width, window.height)
                    layer_cfg.stereo = True
                    layer_cfg.stereo_baseline_mm = 0.  # off-axis rendering already includes the IPD
                    layer_cfg.placement = viz.QuadLayerPlacement(viz.Pose3D((0, 1.5, -1), (1, 0, 0, 0)), window.size)
                    layer = session.add_quad_layer(layer_cfg)
                    # SDK 1.3.132rc1 binds LayerBase.set_visible without a
                    # registered base class: even set_visible(False) raises.
                    # Clear both textures instead; never retain stale imagery.
                    blank = np.zeros((window.height, window.width, 4), np.uint8)
                    blank_buffer, blank_owner = device_buffer(viz, blank)
                    layer.submit(blank_buffer, blank_buffer)
                    self._display_ready.set()
                    print('Wholebody RGB-LiDAR: shared CloudXR tracking + stereo window ready', flush=True)
                    xyz, colors = np.empty((0, 3)), np.empty((0, 3), np.uint8)
                    next_read = 0.
                    capture_stamp = -1.
                    last_status = None
                    while not self._display_stop.is_set() and not session.should_close():
                        frame = session.begin_frame()
                        try:
                            tracking.update()
                            raw = dict(left_controller=controllers.get_left_controller(tracking).data,
                                right_controller=controllers.get_right_controller(tracking).data,
                                head=head.get_head(tracking).data,
                                left_hand=hands.get_left_hand(tracking).data,
                                right_hand=hands.get_right_hand(tracking).data,
                                full_body=body.get_body_pose(tracking).data)
                            with self._display_lock:
                                self._display_latest = raw
                                self._display_stamp = time.monotonic()
                            now = time.monotonic()
                            if now >= next_read:
                                next_read = now+.05
                                try:
                                    xyz, colors, metadata = read_packet(packet_path, max_age_s=max_age_s)
                                    capture_stamp = metadata['capture_monotonic']
                                except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                                    capture_stamp = -1.
                            pose = raw['head']
                            visible = (pose is not None and pose.is_valid and 0 <= now-capture_stamp <= max_age_s)
                            status = ('Anzeige aktiv' if visible else 'Warte auf gültiges Kopftracking' if
                                pose is None or not pose.is_valid else 'Keine frische Punktwolke; Simulation und Laufordner prüfen')
                            if status != last_status:
                                print('RGB-LiDAR: '+status, flush=True)
                                last_status = status
                            if not visible:
                                layer.submit(blank_buffer, blank_buffer)
                            if frame.should_render and visible:
                                p, q = pose.pose.position, pose.pose.orientation
                                position, quaternion = [p.x, p.y, p.z], [q.w, q.x, q.y, q.z]
                                if window.anchor_position is None:
                                    window.anchor(position, quaternion)
                                    layer.set_placement(viz.QuadLayerPlacement(
                                        viz.Pose3D(window.placement(), quaternion), window.size))
                                left, right = window.render(xyz, colors, position, quaternion)
                                # Televiz requires device buffers. Owners stay alive
                                # through end_frame; upload does not touch Isaac's GPU tensors.
                                left_buffer, left_owner = device_buffer(viz, left)
                                right_buffer, right_owner = device_buffer(viz, right)
                                layer.submit(left_buffer, right_buffer)
                        finally:
                            session.end_frame()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._display_error = str(exc)
                print('Wholebody RGB-LiDAR XR stopped: '+str(exc), flush=True)
            finally:
                with self._display_lock:
                    self._display_latest = None
                self._display_ready.set()

        def _get_tracker_data(self):
            with self._display_lock:
                if time.monotonic()-self._display_stamp > .15:
                    return None
                return self._display_latest

        def close(self):
            self._display_stop.set()
            thread = self._display_thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.)
            with self._display_lock:
                self._display_latest = None

    return LidarIsaacTeleopClient
