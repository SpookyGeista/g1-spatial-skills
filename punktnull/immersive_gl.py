"""GPU triangle renderer shared by desktop verification and OpenXR eye views."""
import ctypes
import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader


class EyeFramebuffer:
    """Explicit FBO: hidden-window default buffers are not reliable readback targets."""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.fbo = GL.glGenFramebuffers(1)
        self.color = GL.glGenTextures(1)
        self.depth = GL.glGenRenderbuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.color, 0)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depth)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Incomplete eye framebuffer')

    def bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.width, self.height)

    def pixels(self):
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        data = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, np.uint8).reshape(self.height, self.width, 4)[::-1].copy()

    def close(self):
        GL.glDeleteRenderbuffers(1, [self.depth])
        GL.glDeleteTextures([self.color])
        GL.glDeleteFramebuffers(1, [self.fbo])


class SurfaceRenderer:
    def __init__(self):
        self.program = compileProgram(compileShader('''#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
uniform mat4 mvp;
out vec3 rgb;
void main(){gl_Position=mvp*vec4(position,1);rgb=color;}
''', GL.GL_VERTEX_SHADER), compileShader('''#version 330 core
in vec3 rgb;
out vec4 output_color;
void main(){output_color=vec4(rgb,1);}
''', GL.GL_FRAGMENT_SHADER))
        self.uniform = GL.glGetUniformLocation(self.program, 'mvp')
        self.vao = GL.glGenVertexArrays(1)
        self.vbo, self.ebo = GL.glGenBuffers(2)
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        for index, offset in ((0, 0), (1, 12)):
            GL.glEnableVertexAttribArray(index)
            GL.glVertexAttribPointer(index, 3, GL.GL_FLOAT, False, 24, ctypes.c_void_p(offset))
        GL.glBindVertexArray(0)

    def upload(self, vertices, indices):
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_DYNAMIC_DRAW)
        self.count = int(indices.size)
        GL.glBindVertexArray(0)

    def draw(self, mvp, *, visible=True):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)
        GL.glClearColor(.02, .025, .035, 1.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if visible and self.count:
            GL.glUseProgram(self.program)
            GL.glUniformMatrix4fv(self.uniform, 1, GL.GL_TRUE, np.asarray(mvp, np.float32))
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)

    def close(self):
        GL.glDeleteBuffers(2, [self.vbo, self.ebo])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)
