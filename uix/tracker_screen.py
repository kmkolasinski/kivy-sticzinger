from typing import Tuple

import cv2
from kivy.animation import Animation
from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty

from logging_ops import profile
from uix.base import ProcessingCameraScreen, RectShape


Builder.load_string("""
# kv_start
<TrackerScreen>:
    camera_widget: camera_widget
    tracker_radius_slider: tracker_radius_slider

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
        
        canvas.after:
            Color:
                rgba: 0., 1, 0, 0.5
                
            Line:
                width: 2
                rectangle: self.parent.center_rect

            Color:
                rgba: 1, 0, 0, 1.0
                                
            Line:
                width: 2
                rectangle: self.parent.tracking_rect
          
    MDSlider:
        id: tracker_radius_slider
        min: 10
        max: 80
        value: 20
        top: -100          
            
    MDFloatingActionButton:
        id: take_photo_button
        text: "Capture"
        icon: "camera"
        pos_hint: {"center_x": 0.5, "center_y": 0.15}
        elevation: 8
        on_release:
            self.parent.start_tracking()            

# kv_end
""")


class TrackerScreen(ProcessingCameraScreen):
    tracking_rect = ListProperty([0, 0, 0, 0])
    center_rect = ListProperty([0, 0, 0, 0])
    tracker_radius = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker = None

    def create_center_bbox(self, dsize: Tuple[int, int]) -> RectShape:
        x1, y1 = dsize[0] // 2, dsize[1] // 2
        radius = int(self.tracker_radius_slider.value / 100 * dsize[0])
        bbox = (x1 - radius // 2, y1 - radius // 2, radius, radius)
        return bbox

    def draw_center_rect(self, dsize):
        bbox = self.create_center_bbox(dsize)
        rect = self.bbox_to_canvas_rect(bbox, dsize)
        anim = Animation(center_rect=rect, duration=0.01)
        anim.start(self)

    def start_tracking(self):
        dsize = self.processing_image_size

        bbox = self.create_center_bbox(dsize)
        image = self.get_resized_frame(dsize)

        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(image, bbox)

    @profile()
    def processing_fn_step(self):
        self.update_current_frame()
        if self.current_frame is None:
            return

        if self.is_paused():
            return

        dsize = self.processing_image_size
        self.draw_center_rect(dsize)

        if self.tracker is not None:
            image = self.get_resized_frame(dsize)
            status, bbox = self.tracker.update(image)
            rect = self.bbox_to_canvas_rect(bbox, dsize)
            anim = Animation(tracking_rect=rect, duration=0.01)
            anim.start(self)



