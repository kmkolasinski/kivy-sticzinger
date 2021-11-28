from kivy.lang import Builder

from uix.base import ProcessingCameraScreen

Builder.load_string("""
# kv_start
<TrackerScreen>:
    camera_widget: camera_widget

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
# kv_end
""")


class TrackerScreen(ProcessingCameraScreen):

    def processing_fn_step(self):
        self.update_current_frame()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
