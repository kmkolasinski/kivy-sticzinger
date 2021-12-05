from typing import Optional

import numpy as np
from kivy.animation import Animation
from kivy.clock import mainthread
from kivy.lang import Builder
from kivy.properties import ListProperty, NumericProperty

import matching
import transform
from logging_ops import profile, measuretime
from uix.basic_stitcher import BasicStitcherScreen

Builder.load_string(
    """
# kv_start

<LeftToRightStitcherScreen>:
    camera_widget: camera_widget
    take_photo_button: take_photo_button
    speed_dial_button: speed_dial_button
    num_matches_label: num_matches_label
    
    canvas.after:
        Color:
            rgba: 223/255, 75/255, 73/255, self.line_color_alpha
        Line:
            points: self.line_points
            width: 10. 

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
        # canvas.before:
        #     PushMatrix
        #     Rotate:
        #         angle:-90
        #         origin:self.center
        # canvas.after:
        #     PopMatrix
            
    MDLabel:
        id: num_matches_label
        text: ""
        pos_hint: {"center_x": 0.5, "center_y": 0.2}
        color: 1, 1, 1, 1
            
    MDFloatingActionButton:
        id: take_photo_button
        text: "Capture"
        icon: "camera-plus"
        pos_hint: {"center_x": 0.5, "center_y": 0.15}
        elevation: 8
        on_release:
            self.parent.take_photo_job()        
            
    MDFloatingActionButtonSpeedDial:
        id: speed_dial_button
        data: self.parent.speed_dial_actions
        root_button_anim: True
        pos_hint: {"center_x": 0.8, "center_y": 0.15}
        callback: self.parent.speed_dial_callback
        
        on_open:
            self.parent.pause()
            take_photo_button.disabled = True             
        on_close: 
            self.parent.play()
            take_photo_button.disabled = False      

# kv_end
"""
)


class LeftToRightStitcherScreen(BasicStitcherScreen):

    line_points = ListProperty([(0, 0), (0, 0)])
    line_color_alpha = NumericProperty(0.0)

    @mainthread
    def update_line_guide(self, points: Optional[np.ndarray]):
        if points is not None:
            points = self.camera_widget.to_canvas_coords(points)
            anim = Animation(line_points=points, line_color_alpha=1.0, duration=0.25)
            anim.start(self)
        else:
            anim = Animation(line_color_alpha=0.0, duration=0.1)
            anim.start(self)

    def reset_state(self):
        super(LeftToRightStitcherScreen, self).reset_state()
        self.update_line_guide(None)

    @profile()
    def stitch(self):

        H, matches = self.match_current_photo_with_pano()
        if H is None:
            return False

        stitched_img = self.data["photo"][2]
        current_photo = self.data["current"][2]
        min_matches = self.conf.matching_conf.min_matches.value

        if len(matches) > min_matches:
            with measuretime(f"Glueing images"):

                current_photo = current_photo.copy()
                stitched_img = stitched_img.copy()

                transform.draw_left_to_right_overlap_line(
                    stitched_img, current_photo, H
                )

                new_image = np.concatenate([stitched_img, current_photo], axis=1)
                mask = np.zeros_like(new_image)[:, :, 0]
                height, width = current_photo.shape[:2]
                mask[:, -width:] = 255

            ymin, ymax, xmin, xmax = matching.mask_bbox(mask)
            current_kp, current_des, _ = self.data["current"]

            for k in current_kp:
                k.pt = (k.pt[0] + xmin, k.pt[1] + ymin)

            self.data["pano_proposal"] = (current_kp, current_des, new_image)
            return True
        else:
            self.show_error_snackbar(
                f"Not enough matches ({len(matches)}) required > {min_matches} !"
            )
            return False

    def compute_keypoints_and_matching_info(self):

        data = super(
            LeftToRightStitcherScreen, self
        ).compute_keypoints_and_matching_info()

        H = data[0]

        if H is not None:
            current_pano_image = self.data["photo"][2]
            height, width = current_pano_image.shape[:2]
            line = np.array([[width, 0], [width, height]], dtype=np.float32)
            line = transform.homography_transform(line, H)
            line = line / np.array([self.processing_image_size])
            self.update_line_guide(line)

        return data
