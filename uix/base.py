import time
from abc import abstractmethod, ABC
from typing import Optional, Tuple

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import mainthread
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen

from cameras import CameraWidget
from logging_ops import elapsedtime
from settings import AppSettings

import threading


RectShape = Tuple[int, int, int, int]


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
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

    @property
    def conf(self) -> AppSettings:
        return AppSettings()

    @property
    def max_wait_time(self) -> float:
        return self.conf.stitching_conf.thread_max_wait_time.value

    def on_enter(self, *args):
        print("Entering screen!", self)
        self.processing_thread = StoppableThread(target=self.processing_fn_loop)
        self.processing_thread.start()

    def on_leave(self, *args):
        print("Leaving screen!", self)
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
            with elapsedtime() as dt:
                self.processing_fn_step()

            wait_seconds = max(self.max_wait_time - dt.seconds, 0)
            time.sleep(wait_seconds)

    @abstractmethod
    def processing_fn_step(self):
        pass




class ProcessingCameraScreen(ProcessingScreenBase):
    camera_widget: CameraWidget = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_frame: Optional[np.ndarray] = None

    @mainthread
    def update_current_frame(self):
        self.current_frame = self.camera_widget.get_current_frame()

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
        self.camera_widget.play = True

    def to_normed_coords(self, points: np.ndarray, dsize: Tuple[int, int] = None) -> np.ndarray:
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