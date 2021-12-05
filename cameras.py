from typing import Optional

import cv2
import kivy
import numpy as np
from kivy.clock import mainthread
from kivy.core.camera import Camera as CoreCamera
from kivy.graphics import InstructionGroup, Color, Ellipse
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.uix.image import Image

from logging_ops import profile


class BaseCamera(Camera):
    canvas_points_group: Optional[InstructionGroup] = None
    core_camera: Optional[CoreCamera] = None
    core_camera_cls = CoreCamera

    def get_current_frame(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def get_or_create_canvas_points_group(self):
        if self.canvas_points_group is None:
            self.canvas_points_group = InstructionGroup()
        else:
            self.canvas.remove(self.canvas_points_group)
            self.canvas_points_group.clear()
        return self.canvas_points_group

    @mainthread
    def render_points(self, points: np.ndarray):
        """
        points: [N, 2] camera image normalized (x, y)
        """
        group = self.get_or_create_canvas_points_group()
        points = self.to_canvas_coords(points)
        for x, y in points:
            color = Color(223 / 255, 75 / 255, 73 / 255, 0.5)
            group.add(color)
            rect = Ellipse(pos=(x, y), size=(24, 24))
            group.add(rect)

        self.canvas.add(group)

    def to_canvas_coords(self, points: np.ndarray):
        """
        points: [N, 2] camera image normalized (x, y)
        """
        new_points = []
        sx, sy = self.norm_image_size
        w, h = self.size
        ox, oy = (w - sx) / 2, (h - sy) / 2
        for point in points:
            x, y = point
            y = 1 - y
            x, y = int(x * sx + ox), int(y * sy + oy)
            new_points.append((x, y))
        return new_points

    def _on_index(self, *args):
        self._camera = None
        if self.index < 0:
            return

        if BaseCamera.core_camera is not None:
            self._camera = BaseCamera.core_camera
        else:
            if self.resolution[0] < 0 or self.resolution[1] < 0:
                self._camera = self.core_camera_cls(index=self.index, stopped=True)
            else:
                self._camera = self.core_camera_cls(
                    index=self.index, resolution=self.resolution, stopped=True
                )
            BaseCamera.core_camera = self._camera

        self._camera.bind(on_load=self._camera_loaded)

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
            self._camera.bind(on_texture=self.on_tex)
        else:
            self._camera.stop()


RotatedCameraAndroid = None
if kivy.platform == "android":
    from kivy.core.camera.camera_android import CameraAndroid

    class RotatedCameraAndroid(CameraAndroid):
        @profile()
        def init_camera(self):

            from jnius import autoclass
            from kivy.graphics.texture import Texture
            from kivy.graphics import Fbo, Callback, Rectangle

            Camera = autoclass("android.hardware.Camera")
            SurfaceTexture = autoclass("android.graphics.SurfaceTexture")
            GL_TEXTURE_EXTERNAL_OES = autoclass(
                "android.opengl.GLES11Ext"
            ).GL_TEXTURE_EXTERNAL_OES
            ImageFormat = autoclass("android.graphics.ImageFormat")

            self._release_camera()
            self._android_camera = Camera.open(self._index)
            params = self._android_camera.getParameters()

            width, height = self._resolution

            params.setPreviewSize(height, width)
            params.setFocusMode("continuous-picture")

            self._android_camera.setParameters(params)
            # self._android_camera.setDisplayOrientation()
            self.fps = 30.0

            pf = params.getPreviewFormat()
            assert pf == ImageFormat.NV21  # default format is NV21
            self._bufsize = int(ImageFormat.getBitsPerPixel(pf) / 8.0 * width * height)

            self._camera_texture = Texture(
                width=width,
                height=height,
                target=GL_TEXTURE_EXTERNAL_OES,
                colorfmt="rgba",
            )
            self._surface_texture = SurfaceTexture(int(self._camera_texture.id))
            self._android_camera.setPreviewTexture(self._surface_texture)

            self._fbo = Fbo(size=self._resolution)
            self._fbo["resolution"] = (float(width), float(height))
            self._fbo.shader.fs = """
                #extension GL_OES_EGL_image_external : require
                #ifdef GL_ES
                    precision highp float;
                #endif
    
                /* Outputs from the vertex shader */
                varying vec4 frag_color;
                varying vec2 tex_coord0;
    
                /* uniform texture samplers */
                uniform sampler2D texture0;
                uniform samplerExternalOES texture1;
                uniform vec2 resolution;
    
                void main()
                {
                    vec2 coord = vec2(tex_coord0.y, 1.0 - tex_coord0.x);
                    gl_FragColor = texture2D(texture1, coord);
                }
            """
            with self._fbo:
                self._texture_cb = Callback(lambda instr: self._camera_texture.bind)
                Rectangle(size=self._resolution)

        def decode_frame(self, buf):
            """
            Decode image data from grabbed frame.

            This method depends on OpenCV and NumPy - however it is only used for
            fetching the current frame as a NumPy array, and not required when
            this :class:`CameraAndroid` provider is simply used by a
            :class:`~kivy.uix.camera.Camera` widget.
            """

            w, h = self._resolution
            arr = np.fromstring(buf, "uint8").reshape((w + w // 2, h))
            arr = cv2.cvtColor(arr, 93)  # NV21 -> BGR
            return np.rot90(arr, 3)

        def read_frame(self) -> Optional[np.ndarray]:
            """
            Grab and decode frame in one call
            """
            buffer = self.grab_frame()
            if buffer is not None:
                return self.decode_frame(buffer)
            return buffer


class AndroidCamera(BaseCamera):
    # (640, 480) or (1280, 720) or ...
    camera_resolution = (480, 640)
    core_camera_cls = RotatedCameraAndroid

    def get_current_frame(self) -> Optional[np.ndarray]:
        return self._camera.read_frame()


class LinuxCamera(BaseCamera):
    camera_resolution = (-1, -1)

    def on_tex(self, *args):
        super(LinuxCamera, self).on_tex(*args)

    def frame_from_buf(self):
        if self.texture is None:
            return
        image = np.frombuffer(self.texture.pixels, np.uint8).reshape(
            *self.texture.size[::-1], 4
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_current_frame(self) -> Optional[np.ndarray]:
        return self.frame_from_buf()


# class MockCamera(Camera, BaseCamera):
#     camera_resolution = (640, 480)
#
#     def _on_index(self, *largs):
#         pass
#
#     def get_current_frame(self) -> Optional[np.ndarray]:
#         return self.frame_from_buf()
#
#     def frame_from_buf(self):
#         if self.texture is None:
#             return
#         image = np.frombuffer(self.texture.pixels, np.uint8).reshape(
#             *self.texture.size[::-1], 4
#         )
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = np.flip(image, 0)
#         return image
#
#     def sample_camera(self):
#         image = cv2.imread("assets/images/image-1.jpg")
#         w, h = self.camera_resolution
#         random_crop = get_random_crop(image, h * 3, w * 3)
#         self.random_crop = random_crop
#
#     def _camera_loaded(self, *largs):
#         self.texture = Texture.create(size=self.camera_resolution, colorfmt="rgb")
#         self.texture_size = list(self.texture.size)
#         self.sample_camera()
#
#     def on_tex(self, *l):
#         self.frame_to_screen()
#         super(MockCamera, self).on_tex(*l)
#
#     def frame_to_screen(self):
#
#         w, h = self.camera_resolution
#         image = cv2.resize(self.random_crop, (w, h))
#
#         frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         flipped = np.flip(frame_rgb, 0)
#         buf = flipped.tostring()
#         self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")


class CameraWidget(BaseCamera):
    def __new__(cls, *args, **kwargs) -> BaseCamera:

        if kivy.platform != "linux":
            camera_cls = AndroidCamera
        else:
            camera_cls = LinuxCamera

        camera = camera_cls(
            *args, **kwargs, resolution=camera_cls.camera_resolution, play=False
        )
        return camera


def copy_image_to_texture(img, image: Image):

    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.flip(img, 0)
    image.texture = Texture.create(size=(w, h), colorfmt="rgb")
    image.texture.blit_buffer(img.flatten(), colorfmt="rgb", bufferfmt="ubyte")
