from typing import Optional

import cv2
import kivy
import numpy as np
from kivy.graphics import InstructionGroup, Color, Ellipse

from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.uix.image import Image

from kivy.core.camera import Camera as CoreCamera


class BaseCamera(Camera):
    canvas_points_group: Optional[InstructionGroup] = None
    core_camera: Optional[CoreCamera] = None

    def get_current_frame(self) -> Optional[np.ndarray]:
        pass

    def get_or_create_canvas_points_group(self):
        if self.canvas_points_group is None:
            self.canvas_points_group = InstructionGroup()
        else:
            self.canvas.remove(self.canvas_points_group)
            self.canvas_points_group.clear()
        return self.canvas_points_group

    def render_points(self, points: np.ndarray):
        """
        points: [N, 2] image normalized (x, y)
        """
        group = self.get_or_create_canvas_points_group()

        sx, sy = self.norm_image_size
        w, h = self.size
        ox, oy = (w - sx) / 2, (h - sy) / 2
        for point in points:
            x, y = point
            x, y = int(x * sx + ox), int(y * sy + oy)
            color = Color(223 / 255, 75 / 255, 73 / 255, 0.5)
            group.add(color)
            rect = Ellipse(pos=(x, y), size=(24, 24))
            group.add(rect)

        self.canvas.add(group)

    def to_canvas_coords(self, points: np.ndarray):
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


class AndroidCamera(BaseCamera):
    # (640, 480) or (1280, 720) or
    camera_resolution = (640, 480)
    counter = 0

    # def _on_index(self, *largs):
    #     from kivy.core.camera.camera_android import CameraAndroid
    #     self._camera = None
    #     if self.index < 0:
    #         return
    #     if self.resolution[0] < 0 or self.resolution[1] < 0:
    #         self._camera = CameraAndroid(index=self.index, stopped=True)
    #     else:
    #         self._camera = CameraAndroid(index=self.index,
    #                               resolution=self.resolution, stopped=True)
    #     self._camera.bind(on_load=self._camera_loaded)
    #     if self.play:
    #         self._camera.start()
    #         self._camera.bind(on_texture=self.on_tex)

    def _camera_loaded(self, *largs):
        print(f"Camera loaded, resolution [{self.camera_resolution}] !")
        self.texture = Texture.create(
            size=np.flip(self.camera_resolution), colorfmt="rgb"
        )
        self.texture_size = list(self.texture.size)
        self.frame: Optional[np.ndarray] = None

    def get_current_frame(self) -> Optional[np.ndarray]:
        if hasattr(self, "frame"):
            return self.frame
        return None

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        self.frame = self.frame_from_buf()
        self.frame_to_screen(self.frame)

        super(AndroidCamera, self).on_tex(*l)

    def frame_from_buf(self):
        buffer = self._camera._buffer
        if buffer is None:
            return None

        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), "uint8").reshape(
            (h + h // 2, w)
        )
        frame_bgr = cv2.cvtColor(frame, 93)
        return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tostring()
        self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")


class LinuxCamera(BaseCamera):
    camera_resolution = (-1, -1)

    def _on_index(self, *largs):
        self._camera = None
        if self.index < 0:
            return

        if BaseCamera.core_camera is not None:
            self._camera = BaseCamera.core_camera
        else:
            if self.resolution[0] < 0 or self.resolution[1] < 0:
                self._camera = CoreCamera(index=self.index, stopped=True)
            else:
                self._camera = CoreCamera(
                    index=self.index, resolution=self.resolution, stopped=True
                )
            BaseCamera.core_camera = self._camera

        self._camera.bind(on_load=self._camera_loaded)
        if self.play:
            self._camera.start()
            self._camera.bind(on_texture=self.on_tex)

    def on_tex(self, *l):
        super(LinuxCamera, self).on_tex(*l)

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
            return AndroidCamera(
                *args, **kwargs, resolution=AndroidCamera.camera_resolution
            )
        else:
            return LinuxCamera(
                *args, **kwargs, resolution=LinuxCamera.camera_resolution
            )


def numpy_to_image(img, image: Image):
    h, w, _ = img.shape
    # texture = Texture.create(size=(w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.flip(img, 0)
    image.texture = Texture.create(size=(w, h), colorfmt="rgb")
    image.texture.blit_buffer(img.flatten(), colorfmt="rgb", bufferfmt="ubyte")


def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y : y + crop_height, x : x + crop_width]

    return crop
