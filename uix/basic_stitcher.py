from time import sleep
from typing import Tuple, Optional

import cv2
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.button import MDFloatingActionButton
from kivymd.uix.snackbar import Snackbar

import matching
import transform
from logging_ops import profile, measuretime
from uix.base import ProcessingCameraScreen, RectShape
import numpy as np
import keypoints_extractors as ke_ops
from uix.preview_image import PreviewPanoramaScreen

Builder.load_string(
    """
# kv_start
<BasicStitcherScreen>:
    camera_widget: camera_widget
    take_photo_button: take_photo_button

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
            self.parent.take_photo_job()            
# kv_end
"""
)

STITCHING_NONE = None
STITCHING_INITIALIZED = "STITCHING_INITIALIZED"


class BasicStitcherScreen(ProcessingCameraScreen):
    take_photo_button: MDFloatingActionButton = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = {}
        self.stitching_state = None
        self.photo_index = 0
        self.is_taking_photo = False
        self.preview_window = PreviewPanoramaScreen(name="panorama-preview-screen")

    def reset_state(self):
        self.data = {}
        self.set_stitching_state(STITCHING_NONE)
        self.photo_index = 0
        self.is_taking_photo = False
        self.camera_widget.render_points(np.zeros([0, 2]))

    def set_stitching_state(self, stitching_state: Optional[str] = STITCHING_NONE):
        self.stitching_state = stitching_state
        if stitching_state == STITCHING_NONE:
            self.take_photo_button.icon = "camera-plus"
        else:
            self.take_photo_button.icon = "camera"

    def disable_gui(self, *args):
        self.take_photo_button.disabled = True

    def enable_gui(self, *args):
        self.take_photo_button.disabled = False

    def preview_current_panorama(self):
        self.pause()
        image = self.data["photo"][2]
        self.preview_window.show(
            self.manager, image, self.accept_pano_proposal, self.cancel_preview_screen
        )

    @profile
    def accept_pano_proposal(self,  manager: ScreenManager):
        manager.switch_to(self)
        self.data["photo"] = self.data["pano_proposal"]
        self.photo_index += 1
        # image = self.data["photo"][2].copy()
        # self.save_image_job(image, "stitched.jpg")
        self.play()

    def cancel_preview_screen(self, manager: ScreenManager):
        manager.switch_to(self)
        self.play()

    def take_photo_job(self):
        self.disable_gui()
        Clock.schedule_once(self.take_photo)

    @profile
    def take_photo(self, *args):
        self.is_taking_photo = True

        image = self.current_frame

        if self.stitching_state is None:
            self.set_stitching_state(STITCHING_INITIALIZED)
            self.extract_keypoints("photo", image)
            self.photo_index += 1
        elif self.stitching_state == STITCHING_INITIALIZED:
            status = self.stitch()
            if status:
                self.preview_current_panorama()

        self.enable_gui()
        self.is_taking_photo = False

    @profile
    def processing_fn_step(self):
        self.update_current_frame()
        if self.current_frame is None:
            return

        if not self.is_playing():
            return

        if self.stitching_state == STITCHING_NONE:
            return

        if self.is_taking_photo:
            return

        if self.stitching_state == STITCHING_INITIALIZED:
            status = self.extract_keypoints("current", self.current_frame, log=False)
            self.compute_keypoints_and_matching_info()

    def extract_keypoints(
        self,
        key: str,
        image: np.ndarray,
        resize: bool = True,
        mask: Optional[np.ndarray] = None,
        log: bool = True,
    ) -> bool:

        if image is None:
            return False

        if resize:
            dsize = self.conf.keypoints_extractor_conf.get_image_size()
            image = cv2.resize(image, dsize)
        else:
            dsize = None

        extractor_name = self.conf.keypoints_extractor_conf.keypoint_detector.value
        extractor = ke_ops.create_keypoint_extractor(extractor_name)

        xmin, ymin = 0, 0
        if mask is not None:
            ymin, ymax, xmin, xmax = matching.mask_bbox(mask)
            image = image[ymin:ymax, xmin:xmax]
            mask = mask[ymin:ymax, xmin:xmax]

        with measuretime(
            f"Det Kpts for '{key}'", extra={"shape": image.shape}, log=log
        ):
            keypoints, descriptors = ke_ops.extract_keypoints(
                image, dsize, extractor, mask=mask
            )

            if mask is not None:
                for k in keypoints:
                    k.pt = (k.pt[0] + xmin, k.pt[1] + ymin)

            self.data[key] = (keypoints, descriptors, image)

        return True

    @profile
    def stitch(self, *args):

        if "photo" not in self.data:
            return

        if "current" not in self.data:
            return

        kp1, des1, stitched_img = self.data["photo"]
        kp2, des2, current_photo = self.data["current"]

        with measuretime(
            f"Matching", extra={"num_left_kpt": len(kp1), "num_right:kpt": len(kp2)}
        ):
            H, matches = matching.match_images(
                kp1, des1, kp2, des2, **self.conf.matching_configuration
            )

        with measuretime(f"Drawing Matches", extra={"num_matches": len(matches)}):
            draw_matches = [[m] for m in matches]
            matches_image = cv2.drawMatchesKnn(
                stitched_img,
                kp1,
                current_photo,
                kp2,
                draw_matches,
                flags=2,
                outImg=None,
            )
            self.data["matches_image"] = matches_image

        min_matches = self.conf.matching_conf.min_matches.value

        if len(matches) > min_matches:
            with measuretime(f"Glueing images"):
                current_photo = current_photo.copy()
                stitched_img = stitched_img.copy()
                self.draw_transformed_image_borders(stitched_img, current_photo, H)

                stitched_img, _ = transform.cv_blend_images(
                    current_photo, stitched_img, np.linalg.inv(H)
                )

            self.extract_keypoints("pano_proposal", stitched_img, resize=False)
            return True
        else:
            Snackbar(
                text=f"Not enough matches ({len(matches)}) required > {min_matches} !",
                bg_color=(1, 0, 0, 0.5),
                duration=2,
            ).open()
            return False

    @profile
    def compute_keypoints_and_matching_info(self):

        if self.stitching_state == STITCHING_NONE:
            return None, []

        if "current" not in self.data:
            return None, []

        kp1, des1 = self.data["photo"][:2]
        kp2, des2 = self.data["current"][:2]

        _, matches = matching.match_images(
            kp1, des1, kp2, des2, **self.conf.matching_configuration
        )

        num_matches = len(matches)

        _, matched_points = matching.select_matching_points(kp1, kp2, matches)
        dsize = self.conf.keypoints_extractor_conf.get_image_size()
        normalized_points = self.to_normed_coords(matched_points, dsize)
        normalized_points[:, 1] = 1 - normalized_points[:, 1]
        self.camera_widget.render_points(normalized_points)

        return num_matches, normalized_points

    def draw_transformed_image_borders(
        self, stitched_img: np.ndarray, current_img: np.ndarray, H: np.ndarray
    ):
        height, width = stitched_img.shape[:2]
        lines = [
            ([[0, 0], [width, 0]], (255, 0, 0)),
            ([[width, 0], [width, height]], (0, 255, 0)),
            ([[width, height], [0, height]], (255, 0, 0)),
            ([[0, height], [0, 0]], (0, 255, 0)),
        ]
        line_thickness = 2

        for line, color in lines:
            line = np.array(line, dtype=np.float32)
            line = transform.homography_transform(line, H).astype(np.int32)
            x1, y1 = line[0]
            x2, y2 = line[1]
            cv2.line(current_img, (x1, y1), (x2, y2), color, thickness=line_thickness)
