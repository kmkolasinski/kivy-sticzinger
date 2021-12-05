import threading
import time
from abc import abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import mainthread
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from kivymd.uix.snackbar import Snackbar

from cameras import CameraWidget
from logging_ops import elapsedtime, profile
from settings import AppSettings

RectShape = Tuple[int, int, int, int]


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()


class ProcessingScreenBase(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processing_thread: Optional[StoppableThread] = None
        self.processing_fps = 1

    @property
    def conf(self) -> AppSettings:
        return AppSettings()

    @property
    def max_wait_time(self) -> float:
        return self.conf.stitching_conf.thread_max_wait_time.value

    @property
    def is_auto_next_enabled(self) -> bool:
        return self.conf.stitching_conf.auto_next_image.value

    @profile()
    def on_enter(self, *args):
        self.processing_thread = StoppableThread(target=self.processing_fn_loop)
        self.processing_thread.start()

    @profile()
    def on_leave(self, *args):
        self.processing_thread.stop()
        self.processing_thread = None

    def should_process_frame(self):
        if self.processing_thread is None:
            return False
        if self.processing_thread.stopped():
            return False
        app = App.get_running_app()
        return app is not None and app.running

    def processing_fn_loop(self):
        while self.should_process_frame():
            with elapsedtime() as full_step_dt:
                with elapsedtime() as dt:
                    self.processing_fn_step()

                wait_seconds = max(self.max_wait_time - dt.seconds, 0)
                time.sleep(wait_seconds)

            self.processing_fps = 0.9 * self.processing_fps + 0.1 / full_step_dt.seconds

    @abstractmethod
    def processing_fn_step(self):
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        pass

    @abstractmethod
    def is_paused(self) -> bool:
        pass



class ProcessingCameraScreen(ProcessingScreenBase):
    camera_widget: CameraWidget = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_frame: Optional[np.ndarray] = None

    @mainthread
    def update_current_frame(self):
        self.current_frame = self.camera_widget.get_current_frame()

    def is_playing(self) -> bool:
        return self.camera_widget.play

    def is_paused(self) -> bool:
        return not self.is_playing()

    @profile()
    def play(self):
        self.camera_widget.play = True

    @profile()
    def pause(self):
        self.camera_widget.play = False

    @property
    def processing_image_size(self) -> Tuple[int, int]:
        conf = self.conf.keypoints_extractor_conf
        dsize = conf.get_image_size()
        return dsize

    def get_resized_frame(self, dsize: Tuple[int, int]) -> np.ndarray:
        image = cv2.resize(self.current_frame, dsize)
        return image

    def on_enter(self, *args):
        super(ProcessingCameraScreen, self).on_enter(*args)
        self.play()

    def to_normed_coords(
        self, points: np.ndarray, dsize: Tuple[int, int] = None
    ) -> np.ndarray:
        if dsize is None:
            dsize = self.processing_image_size
        normalized_points = points / np.array([dsize])
        return normalized_points

    def bbox_to_canvas_rect(self, bbox: RectShape, dsize: Tuple[int, int]) -> RectShape:
        x, y, w, h = bbox
        points = np.array([[x, y], [x + w, y + h]])
        # image -> normalized coords
        points = self.to_normed_coords(points, dsize)
        # normed -> camera canvas coords
        (x1, y1), (x2, y2) = self.camera_widget.to_canvas_coords(points)
        rect = (x1, y1, x2 - x1, y2 - y1)
        return rect

    def show_error_snackbar(self, text: str):
        Snackbar(text=text, bg_color=(1, 0, 0, 0.5), duration=2).open()
