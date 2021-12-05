from kivy import Logger
from kivy.lang import Builder

from uix.base import ProcessingCameraScreen

Builder.load_string("""
# kv_start
<CameraViewerScreen>:
    camera_widget: camera_widget

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size

    MDFloatingActionButton:
        id: take_photo_button
        text: "Capture"
        icon: "camera"
        pos_hint: {"center_x": 0.5, "center_y": 0.15}
        elevation: 8
        on_release:
            self.parent.capture_photo()
# kv_end
""")


class CameraViewerScreen(ProcessingCameraScreen):

    def capture_photo(self):
        Logger.info(f"Capturing photo: {self.current_frame.shape}")
        self.show_error_snackbar("I do nothing!")

    def processing_fn_step(self):
        # getting frame from camera
        self.update_current_frame()
