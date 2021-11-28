from kivy.lang import Builder

import keypoints_extractors as ke_ops
from logging_ops import profile
from settings import AppSettings
from uix.base import ProcessingCameraScreen

Builder.load_string(
    """
# kv_start
<KeypointsViewerScreen>:
    camera_widget: camera_widget

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
# kv_end
"""
)


class KeypointsViewerScreen(ProcessingCameraScreen):

    def processing_fn_step(self):
        self.update_current_frame()
        self.detect_and_render_keypoints()

    @profile
    def detect_and_render_keypoints(self):

        image = self.current_frame
        if image is None:
            return

        conf = AppSettings().keypoints_extractor_conf
        dsize = conf.get_image_size()
        extractor_name = conf.keypoint_detector.value
        extractor = ke_ops.create_keypoint_extractor(extractor_name)

        points = ke_ops.detect_screen_keypoints(image, dsize, extractor)
        self.camera_widget.render_points(points)
