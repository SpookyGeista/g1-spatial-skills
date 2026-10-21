"""Real OpenXR projection layers and DeviceIO in one CloudXR session.

No quad layer, virtual screen, guessed eye separation, or head-locked image.
Eye poses/FOV are located for the runtime's predicted display time.
"""
from contextlib import ExitStack
import ctypes as ct
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from .rgbd_surface import read_rgbd, surface_mesh, pose_matrix, projection


def run_immersive(client, packet, max_age_s, replay=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.runtime/immersive'))
    import xr
    import xr.raw_functions
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    from OpenGL import GL
    import vulkan as vk
    import isaacteleop.deviceio as dio
    import isaacteleop.oxr as oxr
    from isaacteleop.cloudxr import CloudXRLauncher
    from .immersive_gl import SurfaceRenderer, EyeFramebuffer
    from .immersive_vulkan import VulkanBridge

    def pointer(handle):
        return ct.cast(handle, ct.c_void_p).value

    with ExitStack() as stack:
        print('Immersiv: CloudXR und echte Stereo-Projektion starten …', flush=True)
        launcher = CloudXRLauncher(install_dir=client._cloudxr_install_dir,
            env_config=client._cloudxr_env_config, accept_eula=True,
            setup_oob=client._use_adb, usb_local=client._use_adb)
        stack.callback(launcher.stop)
        head, hands, controllers, body = dio.HeadTracker(), dio.HandTracker(), dio.ControllerTracker(), dio.FullBodyTrackerPico()
        trackers = [head, hands, controllers, body]
        extensions = list(dict.fromkeys(['XR_KHR_vulkan_enable']+dio.DeviceIOSession.get_required_extensions(trackers)))
        instance = xr.create_instance(xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(application_name='G1 Immersive RGBD', api_version=xr.Version(1, 0, 0)),
            enabled_extension_names=extensions))
        stack.callback(xr.destroy_instance, instance)
        system = xr.get_system(instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        provider = GLFWOffscreenContextProvider()
        stack.callback(provider.destroy)
        graphics = VulkanBridge(instance, system)
        stack.callback(graphics.close)
        session = xr.create_session(instance, xr.SessionCreateInfo(system_id=system,
            next=ct.cast(ct.byref(graphics.binding), ct.c_void_p)))
        stack.callback(xr.destroy_session, session)
        space = xr.create_reference_space(session, xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.STAGE))
        stack.callback(xr.destroy_space, space)
        blend_modes = xr.enumerate_environment_blend_modes(instance, system, xr.ViewConfigurationType.PRIMARY_STEREO)
        blend = xr.EnvironmentBlendMode.OPAQUE if xr.EnvironmentBlendMode.OPAQUE in blend_modes else blend_modes[0]
        formats = xr.enumerate_swapchain_formats(session)
        color_format = vk.VK_FORMAT_R8G8B8A8_UNORM
        if color_format not in formats:
            raise RuntimeError('CloudXR lacks RGBA8 UNORM swapchains')
        swapchains = []
        for config in xr.enumerate_view_configuration_views(instance, system, xr.ViewConfigurationType.PRIMARY_STEREO):
            width, height = config.recommended_image_rect_width, config.recommended_image_rect_height
            factor = min(1., 1024/max(width, height))
            width, height = max(1, int(width*factor)), max(1, int(height*factor))
            swapchain = xr.create_swapchain(session, xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT | xr.SwapchainUsageFlags.TRANSFER_DST_BIT,
                format=color_format, sample_count=1, width=width, height=height, face_count=1, array_size=1, mip_count=1))
            stack.callback(xr.destroy_swapchain, swapchain)
            images = xr.enumerate_swapchain_images(swapchain, element_type=xr.SwapchainImageVulkanKHR)
            framebuffer = EyeFramebuffer(width, height)
            stack.callback(framebuffer.close)
            swapchains.append((swapchain, width, height, images, framebuffer))
        if len(swapchains) != 2:
            raise RuntimeError('Exactly two eye views required')
        handles = oxr.OpenXRSessionHandles(pointer(instance), pointer(session), pointer(space),
            pointer(xr.raw_functions.xrGetInstanceProcAddr))
        tracking = stack.enter_context(dio.DeviceIOSession.run(trackers, handles))
        renderer = SurfaceRenderer()
        stack.callback(renderer.close)
        client._immersive_stats = dict(frames=0, projection_frames=0, triangles=0, fresh_source=False,
                                       headset_image_verified=False, replay=replay)
        client._display_ready.set()
        print('Immersiv bereit: zwei Augenprojektionen, metrische Oberflächen, gemeinsames Körpertracking.', flush=True)
        running, finished = False, False
        stamp, next_read, last_frame = -1., 0., None
        anchor = None
        last_status = None
        while not client._display_stop.is_set() and not finished:
            while True:
                try:
                    event = xr.poll_event(instance)
                except xr.EventUnavailable:
                    break
                if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    changed = ct.cast(ct.byref(event), ct.POINTER(xr.EventDataSessionStateChanged)).contents
                    state = xr.SessionState(changed.state)
                    if state == xr.SessionState.READY:
                        xr.begin_session(session, xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO))
                        running = True
                    elif state == xr.SessionState.STOPPING:
                        xr.end_session(session)
                        running = False
                    elif state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                        finished = True
                elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    finished = True
            if not running or finished:
                client._display_stop.wait(.02)
                continue
            frame = xr.wait_frame(session)
            xr.begin_frame(session)
            layers = []
            try:
                tracking.update()
                raw = dict(left_controller=controllers.get_left_controller(tracking).data,
                    right_controller=controllers.get_right_controller(tracking).data,
                    head=head.get_head(tracking).data, left_hand=hands.get_left_hand(tracking).data,
                    right_hand=hands.get_right_hand(tracking).data, full_body=body.get_body_pose(tracking).data)
                with client._display_lock:
                    client._display_latest, client._display_stamp = raw, time.monotonic()
                now = time.monotonic()
                if now >= next_read:
                    next_read = now+.1
                    try:
                        rgb, depth, meta = read_rgbd(packet, max_age_s=max_age_s, replay=replay)
                        stamp = float(meta['capture_monotonic'])
                        key = (meta['frame_id'], stamp)
                        if key != last_frame:
                            vertices, indices = surface_mesh(rgb, depth, meta['intrinsics'])
                            renderer.upload(vertices, indices)
                            last_frame = key
                            client._immersive_stats['triangles'] = len(indices)//3
                    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                        stamp = -1.
                fresh = stamp >= 0 and (replay or 0 <= time.monotonic()-stamp <= max_age_s)
                client._immersive_stats['fresh_source'] = fresh and not replay
                pose = raw['head']
                valid_head = pose is not None and pose.is_valid
                if anchor is None and valid_head:
                    p, q = pose.pose.position, pose.pose.orientation
                    anchor = pose_matrix([p.x, p.y, p.z], [q.w, q.x, q.y, q.z])
                status = 'AUFZEICHNUNG' if replay else 'Live-Oberfläche' if fresh else 'Warte auf frisches RGB-D-Bild'
                if not valid_head:
                    status += ' · Kopftracking fehlt'
                if status != last_status:
                    print('Immersiv: '+status, flush=True)
                    last_status = status
                if frame.should_render:
                    state, views = xr.locate_views(session, xr.ViewLocateInfo(
                        view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                        display_time=frame.predicted_display_time, space=space))
                    flags = xr.ViewStateFlags.POSITION_VALID_BIT | xr.ViewStateFlags.ORIENTATION_VALID_BIT
                    if state.view_state_flags & flags == flags and len(views) == 2:
                        projection_views = []
                        for view, (swapchain, width, height, images, framebuffer) in zip(views, swapchains):
                            index = xr.acquire_swapchain_image(swapchain)
                            xr.wait_swapchain_image(swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                            try:
                                eye = xr.CompositionLayerProjectionView(pose=view.pose, fov=view.fov,
                                    sub_image=xr.SwapchainSubImage(swapchain=swapchain,
                                        image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(width, height))))
                                framebuffer.bind()
                                p, q, f = view.pose.position, view.pose.orientation, view.fov
                                camera = pose_matrix([p.x, p.y, p.z], [q.w, q.x, q.y, q.z])
                                mvp = projection([f.angle_left, f.angle_right, f.angle_up, f.angle_down]) @ np.linalg.inv(camera) @ (anchor if anchor is not None else np.eye(4))
                                renderer.draw(mvp, visible=fresh and valid_head and anchor is not None)
                                graphics.upload(images[index].image, framebuffer.pixels())
                            finally:
                                xr.release_swapchain_image(swapchain)
                            projection_views.append(eye)
                        layer = xr.CompositionLayerProjection(space=space, views=projection_views)
                        layers = [ct.byref(layer)]
                        client._immersive_stats['projection_frames'] += 1
                client._immersive_stats['frames'] += 1
            finally:
                xr.end_frame(session, xr.FrameEndInfo(display_time=frame.predicted_display_time,
                    environment_blend_mode=blend, layers=layers))
        # Destroying our session is sufficient on application shutdown; do not
        # call xrEndSession while the runtime is not in STOPPING state.
        print('Immersiv beendet: '+str(client._immersive_stats), flush=True)
