"""Vulkan swapchain upload for CloudXR, with explicit portable staging.

This prototype uses GPU render -> host readback -> Vulkan upload. It does not
claim zero-copy or production teleoperation latency.
"""
from contextlib import ExitStack
import ctypes as ct

import vulkan as vk
import xr
from xr.platform import linux as xr_linux


class VulkanBridge:
    def __init__(self, instance, system):
        self.cleanup = ExitStack()
        try:
            self._create(instance, system)
        except Exception:
            self.cleanup.close()
            raise

    def _create(self, instance, system):
        def call(name, *args):
            fn = ct.cast(xr.get_instance_proc_addr(instance, name), getattr(xr, 'PFN_'+name))
            result = xr.check_result(fn(instance, system, *args))
            if result.is_exception():
                raise result

        def extensions(name):
            size = ct.c_uint32()
            call(name, 0, ct.byref(size), None)
            data = ct.create_string_buffer(size.value)
            call(name, size.value, ct.byref(size), data)
            return data.value.decode().split()

        def native(handle, kind):
            return ct.cast(int(vk.ffi.cast('uintptr_t', handle)), kind)

        requirements = xr.GraphicsRequirementsVulkanKHR()
        call('xrGetVulkanGraphicsRequirementsKHR', ct.byref(requirements))
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            pApplicationInfo=vk.VkApplicationInfo(pApplicationName='G1 Immersive', apiVersion=vk.VK_MAKE_VERSION(1, 1, 0)),
            ppEnabledExtensionNames=extensions('xrGetVulkanInstanceExtensionsKHR')), None)
        self.cleanup.callback(vk.vkDestroyInstance, self.instance, None)
        physical = xr_linux.VkPhysicalDevice()
        call('xrGetVulkanGraphicsDeviceKHR', native(self.instance, xr_linux.VkInstance), ct.byref(physical))
        self.physical = vk.ffi.cast('VkPhysicalDevice', ct.cast(physical, ct.c_void_p).value)
        queues = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical)
        self.family = next(i for i, q in enumerate(queues) if q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT)
        self.device = vk.vkCreateDevice(self.physical, vk.VkDeviceCreateInfo(
            pQueueCreateInfos=[vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.family, pQueuePriorities=[1.])],
            ppEnabledExtensionNames=extensions('xrGetVulkanDeviceExtensionsKHR')), None)
        self.cleanup.callback(vk.vkDestroyDevice, self.device, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.family, 0)
        self.binding = xr.GraphicsBindingVulkanKHR(instance=native(self.instance, xr_linux.VkInstance),
            physical_device=physical, device=native(self.device, xr_linux.VkDevice), queue_family_index=self.family, queue_index=0)
        self.pool = vk.vkCreateCommandPool(self.device, vk.VkCommandPoolCreateInfo(
            queueFamilyIndex=self.family, flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT), None)
        self.cleanup.callback(vk.vkDestroyCommandPool, self.device, self.pool, None)
        self.command = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            commandPool=self.pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
        # Eye render targets are capped to 1024x1024 in this prototype.
        self.capacity = 1024*1024*4
        self.buffer = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            size=self.capacity, usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vk.vkGetBufferMemoryRequirements(self.device, self.buffer)
        memory = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)
        flags = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        kind = next(i for i in range(memory.memoryTypeCount) if req.memoryTypeBits & (1 << i)
                    and memory.memoryTypes[i].propertyFlags & flags == flags)
        self.memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(allocationSize=req.size, memoryTypeIndex=kind), None)
        self.cleanup.callback(vk.vkFreeMemory, self.device, self.memory, None)
        self.cleanup.callback(vk.vkDestroyBuffer, self.device, self.buffer, None)
        vk.vkBindBufferMemory(self.device, self.buffer, self.memory, 0)
        self.mapped = vk.vkMapMemory(self.device, self.memory, 0, self.capacity, 0)
        self.cleanup.callback(vk.vkUnmapMemory, self.device, self.memory)

    def upload(self, image_handle, rgba):
        height, width, _ = rgba.shape
        payload = rgba.tobytes()
        if len(payload) > self.capacity:
            raise ValueError('Eye buffer exceeds staging capacity')
        self.mapped[:len(payload)] = payload
        image = vk.ffi.cast('VkImage', ct.cast(image_handle, ct.c_void_p).value)
        vk.vkResetCommandBuffer(self.command, 0)
        vk.vkBeginCommandBuffer(self.command, vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        subrange = vk.VkImageSubresourceRange(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1)
        barrier = vk.VkImageMemoryBarrier(srcAccessMask=0, dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED, newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED, dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image, subresourceRange=subrange)
        vk.vkCmdPipelineBarrier(self.command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [barrier])
        region = vk.VkBufferImageCopy(bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1),
            imageOffset=vk.VkOffset3D(0, 0, 0), imageExtent=vk.VkExtent3D(width, height, 1))
        vk.vkCmdCopyBufferToImage(self.command, self.buffer, image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, [region])
        barrier = vk.VkImageMemoryBarrier(srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, newLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED, dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image, subresourceRange=subrange)
        vk.vkCmdPipelineBarrier(self.command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 1, [barrier])
        vk.vkEndCommandBuffer(self.command)
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[self.command])], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

    def close(self):
        vk.vkDeviceWaitIdle(self.device)
        self.cleanup.close()
