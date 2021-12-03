import datetime
from typing import Optional

import cv2
import numpy as np
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.button import MDFloatingActionButton, MDFloatingBottomButton, MDFloatingActionButtonSpeedDial
from kivymd.uix.label import MDLabel

import keypoints_extractors as ke_ops
import matching
import storage
import transform
from logging_ops import profile, measuretime
from uix.base import ProcessingCameraScreen
from uix.preview_image import PreviewPanoramaScreen

Builder.load_string(
"""
# kv_start
<BasicStitcherScreen>:
    camera_widget: camera_widget
    take_photo_button: take_photo_button
    speed_dial_button: speed_dial_button
    num_matches_label: num_matches_label

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
    
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

STITCHING_NONE = None
STITCHING_INITIALIZED = "STITCHING_INITIALIZED"


class BasicStitcherScreen(ProcessingCameraScreen):
    take_photo_button: MDFloatingActionButton = ObjectProperty()
    speed_dial_button: MDFloatingActionButtonSpeedDial = ObjectProperty()
    num_matches_label: MDLabel = ObjectProperty()

    speed_dial_actions = {
        "Preview": "eye",
        "Finish Current Sequence": "cancel",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = {}
        self.stitching_state = None
        self.photo_index = 0
        self.is_taking_photo = False
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.preview_window = PreviewPanoramaScreen(name="panorama-preview-screen")

    @mainthread
    def reset_state(self):
        self.data = {}
        self.set_stitching_state(STITCHING_NONE)
        self.photo_index = 0
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
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

    def save_image(self, image: np.ndarray, name: str):
        session_id = self.session_id

        def _save(*args):
            with measuretime(f"Saving photo {name}"):
                storage.save_image(image, name, session_id)

        Clock.schedule_once(_save)

    def save_current_frame(self, image: np.ndarray):
        name = f"{self.photo_index}".zfill(3) + ".jpg"
        self.save_image(image, name)

    def preview_pano_proposal(self):
        self.pause()
        image = self.data["pano_proposal"][2]
        self.preview_window.show(
            self.manager, image, self.accept_pano_proposal, self.cancel_preview_screen
        )

    def preview_current_panorama(self):
        self.pause()
        image = self.data["photo"][2]
        self.preview_window.show(
            self.manager, image, None, self.cancel_preview_screen
        )

    @profile
    def accept_pano_proposal(self, manager: ScreenManager):
        manager.switch_to(self)
        self.accept_current_panorama()
        self.play()

    @profile
    def accept_current_panorama(self):
        self.data["photo"] = self.data["pano_proposal"]
        self.photo_index += 1
        self.save_image(self.data["photo"][2], "stitched.jpg")

    def cancel_preview_screen(self, manager: ScreenManager):
        manager.switch_to(self)
        self.play()

    def speed_dial_callback(self, instance: MDFloatingBottomButton):
        if self.stitching_state != STITCHING_INITIALIZED:
            self.show_error_snackbar(f"Start photos sequence first !")
            return

        if instance.icon == "cancel":
            self.reset_state()
            self.speed_dial_button.close_stack()
        elif instance.icon == "eye":
            self.preview_current_panorama()

    def take_photo_job(self):
        self.disable_gui()
        Clock.schedule_once(self.take_photo)

    @profile
    def take_photo(self, *args):
        self.is_taking_photo = True

        image = self.current_frame
        self.save_current_frame(image)

        if self.stitching_state is None:
            self.set_stitching_state(STITCHING_INITIALIZED)
            self.extract_keypoints("photo", image)
            self.photo_index += 1
        elif self.stitching_state == STITCHING_INITIALIZED:
            status = self.stitch()
            if status:
                if self.is_auto_next_enabled:
                    self.accept_current_panorama()
                else:
                    self.preview_pano_proposal()

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
            if status:
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
                transform.draw_transformed_image_borders(stitched_img, current_photo, H)

                stitched_img, _ = transform.cv_blend_images(
                    current_photo, stitched_img, np.linalg.inv(H)
                )

            self.extract_keypoints("pano_proposal", stitched_img, resize=False)
            return True
        else:
            self.show_error_snackbar(
                f"Not enough matches ({len(matches)}) required > {min_matches} !"
            )
            return False

    def match_current_photo_with_pano(self):

        if "photo" not in self.data:
            return None, []

        if "current" not in self.data:
            return None, []

        kp1, des1, stitched_img = self.data["photo"]
        kp2, des2, current_photo = self.data["current"]

        with measuretime(
                f"Matching", extra={"num_left_kpt": len(kp1), "num_right:kpt": len(kp2)}
        ):
            H, matches = matching.match_images(
                kp1, des1, kp2, des2, **self.conf.matching_configuration
            )
        return H, matches

    @profile
    def compute_keypoints_and_matching_info(self):

        if self.stitching_state == STITCHING_NONE:
            return None, None, []

        if "current" not in self.data:
            return None, None, []

        kp1, des1 = self.data["photo"][:2]
        kp2, des2 = self.data["current"][:2]

        H, matches = matching.match_images(
            kp1, des1, kp2, des2, **self.conf.matching_configuration
        )

        min_matches = self.conf.matching_conf.min_matches.value
        self.num_matches_label.text = f"{len(matches)} / {min_matches}"

        _, matched_points = matching.select_matching_points(kp1, kp2, matches)
        dsize = self.conf.keypoints_extractor_conf.get_image_size()
        normalized_points = self.to_normed_coords(matched_points, dsize)
        self.camera_widget.render_points(normalized_points)

        return H, matches, normalized_points
