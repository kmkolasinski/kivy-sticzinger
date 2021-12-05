import datetime
import json
import threading
import time
from typing import Optional, Tuple, Any, Dict

import cv2
import kivy
import numpy as np
from kivy import Logger
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, InstructionGroup
from kivy.lang import Builder
from kivy.logger import LoggerHistory
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty,
    ListProperty,
    ConfigParser,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.settings import Settings, SettingsWithNoMenu
from kivy.uix.widget import Widget
from kivy.utils import escape_markup
from kivymd.app import MDApp
from kivymd.uix.bottomsheet import MDGridBottomSheet
from kivymd.uix.button import (
    MDFloatingActionButton,
    MDFloatingBottomButton,
    MDFloatingActionButtonSpeedDial,
)
from kivymd.uix.label import MDLabel
from kivymd.uix.navigationdrawer import MDNavigationDrawer
from kivymd.uix.snackbar import Snackbar

import cameras
import matching
import storage
import transform
from cameras import CameraWidget
from logging_ops import measuretime

pos_hint = {"center_x": 0.5, "center_y": 0.5}


camera_canvas = """
<ScoreGauge>
    canvas:
        Color:
            rgba: 200/255, 200/255, 200/255, .5
        Line:
            width: 6.
            circle:
                (self.center_x, self.center_y, min(self.width, self.height)
                / 2, 0, 360, 50)
    
        Color:
            rgba: 33/255, 150/255, 243/255, .8
        Line:
            width: 5.
            circle:
                (self.center_x, self.center_y, min(self.width, self.height)
                / 2, 0, self.score, 50)
    MDLabel:
        center: root.center
        text: self.parent.score_text
        color: 'black'
        bold: True
        halign: 'center'
        font_style: "H6"


<NumMatchesGauge>:
    canvas:
        Color:
            rgba: 243/255, 0/255, 0/255, 0.5
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [(2, 2), (2, 2), (2, 2), (2, 2)]
    MDLabel:
        center: self.parent.center
        size_hint: (None, None)
        halign: 'center'
        text: "{:.0f}".format(self.parent.num_matches)
        bold: True

<CameraCanvas>:
    camera_widget: camera_widget
    logger_history: logger_history
    num_matches_gauge: num_matches_gauge
    
    canvas.before:
        Color:
            rgba: 0/255, 0/255, 0/255, 0.1
        Rectangle:
            pos: self.pos
            size: self.size    
        
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

    NumMatchesGauge:
        id: num_matches_gauge
        size: root.width / 4, 100
        center: self.parent.center[0], root.top - 250
            
    MDLabel:
        id: logger_history
        text: ""
        font_size: 20
        bold: False
        color: 1, 1, 1, 1
        top: self.parent.top - 200
        size: self.parent.size[0], 150
    
"""


class ScoreGauge(Widget):
    score = NumericProperty(0)
    score_text = StringProperty("0")


class NumMatchesGauge(Widget):
    num_matches = NumericProperty(0)

    def update_num_matches(self, num_matches: int):
        anim = Animation(num_matches=int(num_matches), duration=0.1)
        anim.start(self)


class CameraCanvas(Widget):
    camera_widget: CameraWidget = ObjectProperty(None)
    logger_history: MDLabel = ObjectProperty(None)
    num_matches_gauge: NumMatchesGauge = ObjectProperty(None)
    matched_points = ListProperty([])
    group: Optional[InstructionGroup] = None
    line_points = ListProperty([(0, 0), (0, 0)])
    line_color_alpha = NumericProperty(1.0)

    def update_overlay(self, num_matches: int, points: np.ndarray, line: Optional[np.ndarray]):

        self.num_matches_gauge.update_num_matches(num_matches)

        # lines = []
        # for entry in LoggerHistory.history[:10]:
        #     lines.append(entry.msg)
        # lines = "\n".join(lines)
        # self.logger_history.text = lines

        if self.group is None:
            self.group = InstructionGroup()
        else:
            self.canvas.remove(self.group)
            self.group.clear()

        sx, sy = self.camera_widget.norm_image_size
        w, h = self.size
        ox, oy = (w - sx) / 2, (h - sy) / 2
        for point in points:
            x, y = point
            x, y = int(x * sx + ox), int(y * sy + oy)
            color = Color(223 / 255, 75 / 255, 73 / 255, 0.5)
            self.group.add(color)
            rect = Ellipse(pos=(x, y), size=(24, 24))
            self.group.add(rect)

        self.canvas.add(self.group)

        if line is not None:
            line_points = []
            for point in line:
                x, y = point
                x, y = int(x * sx + ox), int(y * sy + oy)
                y = min(max(y, oy), oy + sy)
                x = min(max(x, 0), w)
                line_points.append((x, y))

            anim = Animation(line_points=line_points, line_color_alpha=1.0, duration=0.25)
            anim.start(self)
        else:
            anim = Animation(line_color_alpha=0.0, duration=0.01)
            anim.start(self)


Builder.load_string(camera_canvas)


class PreviewPanoramaWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        label = MDLabel(text="")
        self.return_btn = MDFloatingActionButton(
            text="Return",
            icon="cancel",
            pos_hint={"center_x": 0.3, "center_y": 0.1},
            elevation=8,
        )

        self.accept_btn = MDFloatingActionButton(
            text="Accept",
            icon="check",
            pos_hint={"center_x": 0.7, "center_y": 0.1},
            elevation=8,
        )

        render = Image()

        scroll_view = ScrollView(do_scroll_x=False, do_scroll_y=False)
        scatter = Scatter(do_rotation=False)
        scatter.add_widget(render)
        scatter.scale = 5
        scroll_view.add_widget(scatter)
        self.render = render

        self.add_widget(label)
        self.add_widget(scroll_view)
        self.add_widget(self.return_btn)
        self.add_widget(self.accept_btn)

    def set_image(self, image):
        cameras.copy_image_to_texture(image, self.render)

    def enable_accept(self, toggle: bool = True):
        if toggle and hasattr(self.accept_btn, "saved_pos_hint"):
            self.accept_btn.pos_hint = self.accept_btn.saved_pos_hint
            return
        if not toggle:
            self.accept_btn.saved_pos_hint = self.accept_btn.pos_hint
            self.accept_btn.pos_hint = {"center_x": 0.0, "center_y": -0.1}


class MainWindow(Screen):
    stop = threading.Event()
    screen_manager: ScreenManager = ObjectProperty()
    nav_drawer = ObjectProperty()
    screen_pano: PreviewPanoramaWindow = ObjectProperty()
    take_photo_button: MDFloatingActionButton = ObjectProperty()
    speed_dial_button: MDFloatingActionButtonSpeedDial = ObjectProperty()

    speed_dial_actions = {
        "Debug Images": "eye",
        "Debug Matches": "ab-testing",
        "Finish current Sequence": "cancel",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data = {}
        self.frame: np.ndarray = None
        self.taking_photo = False
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.photo_index = 0
        self.stitching_mode = None
        self.frame_monitor_thread = threading.Thread(target=self.infinite_loop)

        Clock.schedule_once(self.bind_pano_actions)
        Clock.schedule_interval(self.snapshot_frame, 0.25)

    @property
    def config(self) -> ConfigParser:
        return ConfigParser.get_configparser("app")

    @property
    def auto_next_image(self) -> bool:
        value = self.config.get("Settings", "auto_next_image")
        return value == "1"

    @property
    def CV_IMAGE_SIZE(self) -> Tuple[int, int]:
        value = self.config.get("Settings", "preview_matching_size")
        value = int(value)
        return value, value

    @property
    def fe(self):
        value = self.config.get("Settings", "keypoint_detector").upper()
        if value == "SIFT":
            return cv2.SIFT_create()
        elif value == "BRISK":
            return cv2.BRISK_create()
        elif value == "KAZE":
            return cv2.KAZE_create(extended=False)
        elif value == "KAZE_EXTENDED":
            return cv2.KAZE_create(extended=True)
        elif value == "ORB_FAST":
            return cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
        elif value == "ORB_HARRIS":
            return cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
        elif value == "ORB_FAST_512":
            return cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_FAST_SCORE)
        elif value == "ORB_FAST_1024":
            return cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_FAST_SCORE)
        elif value == "ORB_HARRIS_512":
            return cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_HARRIS_SCORE)
        elif value == "ORB_HARRIS_1024":
            return cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_HARRIS_SCORE)
        else:
            Logger.warning(f"Invalid detector type : {value}, using SIFT")
            return cv2.SIFT_create()

    @property
    def min_matches(self) -> int:
        value = self.config.get("Settings", "min_matches")
        return int(value)

    @property
    def matching_configuration(self) -> Dict[str, Any]:
        key = "Matcher Settings"
        cross_check = self.config.get(key, "cross_check") == "1"

        detector_type = self.config.get("Settings", "keypoint_detector").upper()

        if "SIFT" in detector_type or "KAZE" in detector_type:
            bf_matcher_norm = "NORM_L2"
        else:
            bf_matcher_norm = "NORM_HAMMING"

        conf = {
            "bf_matcher_cross_check": cross_check,
            "bf_matcher_norm": bf_matcher_norm,
            "lowe": float(self.config.get(key, "lowe_ratio")),
            "ransack_threshold": float(self.config.get(key, "ransack_threshold")),
            "matcher_type": self.config.get(key, "matcher_type"),
        }
        Logger.info(f"Matching configuration: {conf}")
        return conf

    def monitor_photo(self):
        self.frame_monitor_thread.start()

    def bind_pano_actions(self, *args):
        # https://stackoverflow.com/questions/48963808/binding-kivy-objectproperty-to-a-child-widget-doesnt-seem-to-work-outside-of-ro
        self.screen_pano.return_btn.bind(on_release=self.cancel_pano)
        self.screen_pano.accept_btn.bind(on_release=self.accept_pano)
        self.monitor_photo()

    def speed_dial_callback(self, instance: MDFloatingBottomButton):
        if self.stitching_mode is None:
            Snackbar(
                text=f"Start photos sequence first !",
                bg_color=(1, 0, 0, 0.5),
                duration=2,
            ).open()
            return

        if instance.icon == "eye":
            self.preview_pano()
        elif instance.icon == "ab-testing":
            self.preview_matches()
        elif instance.icon == "cancel":
            self.cancel_photo()
            self.speed_dial_button.close_stack()

    def maybe_show_select_stitching_mode(self):
        if self.stitching_mode is not None:
            return False

        bottom_sheet_menu = MDGridBottomSheet()
        data = {
            ("Start Default Capture", "DEFAULT"): "arrow-expand-all",
            ("Start Left to Right Capture", "LEFT_TO_RIGHT"): "arrow-expand-right",
            ("Close", None): "cancel",
        }
        for item in data.items():
            bottom_sheet_menu.add_item(
                item[0][0],
                lambda x, y=item[0][1]: self.set_stitching_mode(y),
                icon_src=item[1],
            )
        bottom_sheet_menu.open()
        return True

    @property
    def sm(self):
        return self.screen_manager

    def set_stitching_mode(self, stitching_mode: str):
        self.stitching_mode = stitching_mode
        if stitching_mode is None:
            self.take_photo_button.icon = "camera-plus"
        else:
            self.take_photo_button.icon = "camera"

    def cancel_pano(self, *args):
        self.sm.current = "camera"
        self.play()
        self.speed_dial_button.close_stack()

    def play(self):
        self.camera.play = True

    def pause(self):
        self.camera.play = False

    def accept_pano(self, *args):
        with measuretime(f"Saving panorama"):
            self.data["photo"] = self.data["pano_proposal"]
            self.screen_pano.enable_accept(True)
            self.sm.current = "camera"
            image = self.data["photo"][2].copy()

            self.photo_index += 1
            self.save_image_job(image, "stitched.jpg")
            self.play()

    def on_photo_captured(self, *args):
        photo = self.data["pano_proposal"][2]
        self.screen_pano.set_image(photo)
        self.screen_pano.enable_accept(True)
        self.sm.current = "pano"
        self.pause()

    def cancel_photo(self, *args):
        if self.stitching_mode is None:
            return
        self.screen_pano.enable_accept(True)
        self.data = {}
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.photo_index = 0
        self.set_stitching_mode(None)

    def preview_pano(self, *args):
        if "photo" in self.data:
            self.data["pano_proposal"] = self.data["photo"]
            keypoints, des, image = self.data["photo"]
            image = image.copy()
            self.camera.play = False
            cv2.drawKeypoints(image, keypoints, image, color=(255, 0, 0))
            self.screen_pano.set_image(image)
            self.screen_pano.enable_accept(False)
            self.sm.current = "pano"

    def preview_matches(self, *args):
        self.take_photo(match_mode=True)
        if "matches_image" in self.data:
            self.camera.play = False
            matches_image = self.data["matches_image"]
            self.screen_pano.set_image(matches_image)
            self.screen_pano.enable_accept(False)
            self.sm.current = "pano"

    @property
    def camera(self) -> CameraWidget:
        return self.camera_canvas.camera_widget

    def snapshot_frame(self, *args):
        # with measuretime("Get Frame"):
        self.frame = self.camera.get_current_frame()

    def analyse_photo(
        self, key: str, image: np.ndarray, resize: bool = True, mask=None, log=True
    ):

        if image is None:
            return False

        if resize:
            image = cv2.resize(image, self.CV_IMAGE_SIZE)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if mask is not None:
            ymin, ymax, xmin, xmax = matching.mask_bbox(mask)
            gray = gray[ymin:ymax, xmin:xmax]
            mask = mask[ymin:ymax, xmin:xmax]
            # Logger.info(f"Using masked image for keypoint detection: {gray.shape}")

        with measuretime(f"Det Kpts for '{key}'", extra={"shape": gray.shape}, log=log):

            keypoints, des = self.fe.detectAndCompute(gray, mask=mask)

            if mask is not None:
                for k in keypoints:
                    k.pt = (k.pt[0] + xmin, k.pt[1] + ymin)

            self.data[key] = (keypoints, des, image)

        return True

    def stitch(self, *args):
        if self.stitching_mode == "LEFT_TO_RIGHT":
            return self.stitch_left_right()
        else:
            return self.stitch_default_method()

    def stitch_left_right(self, *args):
        Logger.info("Running Stitching images")
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
                kp1, des1, kp2, des2, **self.matching_configuration
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

        if len(matches) > self.min_matches:
            with measuretime(f"Glueing images"):

                current_photo = current_photo.copy()
                stitched_img = stitched_img.copy()

                height, width = stitched_img.shape[:2]

                line = np.array([[width, 0], [width, height]], dtype=np.float32)
                line = transform.homography_transform(line, H).astype(np.int32)
                x1, y1 = line[0]
                x2, y2 = line[1]
                line_thickness = 2
                cv2.line(
                    current_photo,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    thickness=line_thickness,
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
            Snackbar(
                text=f"Not enough matches ({len(matches)}) required > {self.min_matches} !",
                bg_color=(1, 0, 0, 0.5),
                duration=2,
            ).open()
            return False

    def stitch_default_method(self, *args):
        Logger.info("Running Stitching images - top/bottom")
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
                kp1, des1, kp2, des2, **self.matching_configuration
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

        if len(matches) > self.min_matches:
            with measuretime(f"Glueing images"):
                current_photo = current_photo.copy()
                stitched_img = stitched_img.copy()
                self.draw_transformed_image_borders(stitched_img, current_photo, H)

            stitched_img, _ = transform.cv_blend_images(
                current_photo, stitched_img, np.linalg.inv(H)
            )
            self.analyse_photo("pano_proposal", stitched_img, resize=False)

            return True
        else:
            Snackbar(
                text=f"Not enough matches ({len(matches)}) required > {self.min_matches} !",
                bg_color=(1, 0, 0, 0.5),
                duration=2,
            ).open()
            return False

    def draw_transformed_image_borders(self, stitched_img, current_img, H):
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

    def take_photo_job(self, *args):
        Clock.schedule_once(lambda _: self.take_photo())

    def save_image_job(self, image, name: str):
        session_id = self.session_id

        def _save(*args):
            with measuretime(f"Saving photo {name}"):
                storage.save_image(image, name, session_id)

        Clock.schedule_once(_save)

    def take_photo(self, match_mode: bool = False, *args):

        if self.taking_photo:
            return

        if match_mode and "photo" not in self.data:
            return

        if self.maybe_show_select_stitching_mode():
            self.take_photo_button.disabled = False
            return

        self.take_photo_button.disabled = True
        self.taking_photo = True

        with measuretime("Taking photo"):
            image = self.camera.get_current_frame()

            if not match_mode:
                name = f"{self.photo_index}".zfill(3) + ".jpg"
                self.save_image_job(image, name)

            if "photo" not in self.data:
                self.photo_index += 1
                self.analyse_photo("photo", image)
            else:
                status = self.stitch()
                if not match_mode and status:
                    if self.auto_next_image:
                        self.accept_pano()
                    else:
                        self.on_photo_captured()

        self.taking_photo = False
        self.take_photo_button.disabled = False

    def infinite_loop(self):
        iteration = 0
        while True:
            start_time = time.perf_counter()
            if self.stop.is_set():
                # Stop running this thread so the main Python process can exit.
                return

            if not self.camera.play or self.taking_photo:
                time.sleep(0.25)
                continue

            iteration += 1

            status = self.analyse_photo("current", self.frame, log=False)
            num_matches, points, line = self.match_score()
            if not status or num_matches is None:
                self.camera_canvas.update_overlay(0, [], line)
            else:
                self.camera_canvas.update_overlay(num_matches, points, line)

            end_time = time.perf_counter()
            delta = end_time - start_time
            wait_time = max(0.25 - delta, 0)
            time.sleep(wait_time)

    def match_score(self):

        if self.stitching_mode == "LEFT_TO_RIGHT":
            # left to right
            line = np.array([[0, 0], [0, self.CV_IMAGE_SIZE[1]]], dtype=np.float32)
        else:
            # default mode
            line = None

        if "photo" not in self.data:
            return None, [], line

        if "current" not in self.data:
            return None, [], line

        with measuretime(f"Matching Frame", log=False) as dt:
            current_pano_image = self.data["photo"][2]
            kp1, des1 = self.data["photo"][:2]
            kp2, des2 = self.data["current"][:2]

            H, matches = matching.match_images(
                kp1, des1, kp2, des2, **self.matching_configuration
            )

        Logger.info(
            f"Matching took={dt.t:.3f} [s] NM={len(matches)} NK1={len(kp1)} NK2={len(kp2)}"
        )

        num_matches = len(matches)

        _, current_photo_points = matching.select_matching_points(kp1, kp2, matches)
        normalized_points = current_photo_points / np.array([self.CV_IMAGE_SIZE])

        # inverting Y coordinates
        normalized_points[:, 1] = 1 - normalized_points[:, 1]

        height, width = current_pano_image.shape[:2]

        if H is not None:
            if self.stitching_mode == "LEFT_TO_RIGHT":
                line = np.array([[width, 0], [width, height]], dtype=np.float32)
                line = transform.homography_transform(line, H)

                line = line / np.array([self.CV_IMAGE_SIZE])
                line[:, 1] = 1 - line[:, 1]

        return num_matches, normalized_points, line


main_app_uix = """
<ContentNavigationDrawer>:    
    ScrollView:        
        MDList:            
            OneLineListItem:
                text: "Settings"
                on_release:
                    root.camera_screen.pause() 
                    app.open_settings()
                                    
            OneLineListItem:
                text: "Camera"
                on_release:
                    root.nav_drawer.set_state("close")
                    root.camera_screen.play() 
                    root.screen_manager.current = "camera"
            
            OneLineListItem:
                text: "Logs"
                on_release:
                    root.nav_drawer.set_state("close")
                    root.camera_screen.pause() 
                    app.show_logger_history()
    

<MainWindow>:
    camera_canvas: camera_canvas
    speed_dial_button: speed_dial_button
    take_photo_button: take_photo_button
    
    CameraCanvas:
        id: camera_canvas
    
    MDFloatingActionButton:
        id: take_photo_button
        text: "Capture"
        icon: "camera-plus"
        pos_hint: {"center_x": 0.5, "center_y": 0.15}
        elevation: 8
        on_release:
            self.disabled = True
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
            
            
<MainApp>:
    nav_drawer: nav_drawer
    screen_manager: screen_manager
    camera_screen: camera_screen
    
    MDToolbar:
        id: toolbar
        pos_hint: {"top": 1}
        elevation: 10
        title: "Sticzinger"
        left_action_items: [["menu", app.open_navigation_drawer]]

    MDNavigationLayout:
        x: toolbar.height

        ScreenManager:
            id: screen_manager
            
            Screen:
                name: "camera"                                
                MainWindow:
                    id: camera_screen
                    screen_pano: screen_pano
                    screen_manager: screen_manager
                    nav_drawer: nav_drawer
            
            PreviewPanoramaWindow:
                id: screen_pano
                name: "pano"

            Screen:
                logs: logs
                name: "logs"
                                
                ScrollView: 
                    do_scroll_x: False
                    do_scroll_y: True
                    top: -toolbar.height / 2
                    size_hint: (1, None)
                    size: (Window.width, Window.height)

                    MDLabel:
                        markup: True
                        id: logs
                        size_hint_y: None
                        font_size: 20
                        height: self.texture_size[1]
                        text_size: self.width, None
                        padding: 5, 5
                        text: ""
     
        MDNavigationDrawer:
            id: nav_drawer
            
            ContentNavigationDrawer:
                screen_manager: screen_manager
                nav_drawer: nav_drawer
                camera_screen: camera_screen

"""


class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()


class MainApp(Screen):
    camera_screen: MainWindow = ObjectProperty()
    nav_drawer: MDNavigationDrawer = ObjectProperty()


Builder.load_string(main_app_uix)


if kivy.platform == "linux":
    from kivy.core.window import Window

    Window.size = (480, 800)
    Window.top = 50
    Window.left = 1600


settings_json = json.dumps(
    [
        {
            "key": "auto_next_image",
            "type": "bool",
            "title": "Auto next image",
            "desc": "Skip panorama preview after taking photo",
            "section": "Settings",
        },
        {
            "key": "keypoint_detector",
            "type": "options",
            "title": "Keypoint Detector",
            "desc": "OpenCV keypoint detector type. Affects the speed of application",
            "options": [
                "SIFT",
                "BRISK",
                "KAZE",
                "KAZE_EXTENDED",
                "ORB_FAST",
                "ORB_HARRIS",
                "ORB_FAST_512",
                "ORB_FAST_1024",
                "ORB_HARRIS_512",
                "ORB_HARRIS_1024",
            ],
            "section": "Settings",
        },
        {
            "key": "min_matches",
            "type": "options",
            "title": "Keypoint Detector Min Matches",
            "desc": "minimum matches to accept photo",
            "options": ["20", "30", "40", "50"],
            "section": "Settings",
        },
        {
            "key": "preview_matching_size",
            "type": "options",
            "title": "Preview Image Size",
            "desc": "Image size used to stitch images, the lower the faster.",
            "options": ["300", "350", "400", "500", "600"],
            "section": "Settings",
        },
        {
            "key": "cross_check",
            "type": "bool",
            "title": "MF Matcher Cross Check option",
            "desc": "Applicable only to BFMatcher",
            "section": "Matcher Settings",
        },
        {
            "key": "matcher_type",
            "type": "options",
            "title": "Matcher Type",
            "desc": "OpenCV matcher type",
            "options": ["brute_force", "flann"],
            "section": "Matcher Settings",
        },
        {
            "key": "lowe_ratio",
            "type": "options",
            "title": "Lowe's parameter",
            "desc": "Applicable only when Matcher is Flann or cross_check is set to False",
            "options": ["0.5", "0.6", "0.7", "0.8", "0.9"],
            "section": "Matcher Settings",
        },
        {
            "key": "ransack_threshold",
            "type": "options",
            "title": "RANSACK threshold",
            "desc": "Reprojection error threshold",
            "options": ["5", "7", "10", "14"],
            "section": "Matcher Settings",
        },
    ]
)


class MyApp(MDApp):
    def build(self):

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        self.settings_cls = SettingsWithNoMenu
        self.use_kivy_settings = True
        self.mainApp = MainApp()
        self.mainApp.nav_drawer.bind(state=self.open_navigation_drawer_state_change)

        return self.mainApp

    def on_stop(self):
        print("Exiting!")
        self.mainApp.camera_screen.stop.set()
        super(MyApp, self).on_stop()

    # def on_pause(self):
    #     print("Pausing APP!")
    #     if hasattr(self, "mainApp"):
    #         self.mainApp.camera_screen.pause()
    #     return super(MyApp, self).on_pause()
    #
    # def on_resume(self):
    #     print("Resuming APP!")
    #     if hasattr(self, "mainApp"):
    #         self.mainApp.camera_screen.play()
    #     return super(MyApp, self).on_resume()

    def open_navigation_drawer(self, instance):
        self.mainApp.camera_screen.pause()
        self.mainApp.nav_drawer.set_state("open")

    def open_navigation_drawer_state_change(self, instance, value):
        screen_name = self.mainApp.screen_manager.current
        if value == "close" and screen_name == "camera":
            self.mainApp.camera_screen.play()

    def show_logger_history(self):
        self.root.screen_manager.current = "logs"
        logs_label = self.root.screen_manager.get_screen("logs").logs
        logs_label.text = self.get_logger_history()

    def get_logger_history(self) -> str:
        lines = []
        for entry in LoggerHistory.history[::-1]:
            lines.append(entry.msg)
        text = "\n".join(lines)
        text = escape_markup(text)
        return f"[font=assets/fonts/VeraMono.ttf]{text}[/font]"

    def build_config(self, config: ConfigParser):
        config.setdefaults(
            "Settings",
            {
                "auto_next_image": "0",
                "keypoint_detector": "ORB_HARRIS",
                "min_matches": "30",
                "preview_matching_size": "500",
            },
        )

        config.setdefaults(
            "Matcher Settings",
            {
                "cross_check": "1",
                "matcher_type": "brute_force",
                "lowe_ratio": "0.7",
                "ransack_threshold": "7",
            },
        )

    def build_settings(self, settings: Settings):
        settings.add_json_panel("Settings", self.config, data=settings_json)

    def on_config_change(self, config, section, key, value):
        print("Config changed")
        if key in ("keypoint_detector", "stitching_mode", "preview_matching_size"):
            self.mainApp.camera_screen.cancel_photo()


if __name__ == "__main__":

    from kivy.utils import platform

    if platform == "android":
        from android.permissions import request_permissions, Permission

        request_permissions(
            [Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        )

    MyApp().run()
