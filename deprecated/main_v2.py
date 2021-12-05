import datetime
import os
import sys
import threading
import time
from typing import Optional

import cv2
import kivy
import numpy as np
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, InstructionGroup
from kivy.lang import Builder
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty,
    ListProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.settings import SettingsWithSidebar
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivymd.uix.button import MDFloatingActionButton
from kivymd.uix.label import MDLabel
from kivymd.uix.snackbar import Snackbar

import cameras
import matching
import rectangle_ops
import transform
import storage
from cameras import CameraWidget

pos_hint = {"center_x": 0.5, "center_y": 0.5}
CV_IMAGE_SIZE = (300, 300)

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


<OverlapGauge>:
    canvas:
        Color:
            rgba: 33/255, 150/255, 243/255, 0.5
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [(10, 10), (10, 10), (10, 10), (10, 10)]
    MDLabel:
        center: self.parent.center
        size_hint: (None, None)
        halign: 'center'
        text: "{:.2f}".format(self.parent.overlap)
        bold: True

<CameraCanvas>:
    camera_widget: camera_widget
    # score_gauge: score_gauge
    # overlap_gauge: overlap_gauge
    
    canvas.before:
        Color:
            rgba: 33/255, 150/255, 243/255, 0.1
        Rectangle:
            pos: self.pos
            size: self.size    
        
    canvas.after:
        Color:
            rgba: 33/255, 150/255, 243/255, 0.8
        Line:
            points: self.line_points
            width: 10.
            
    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
    
"""


class ScoreGauge(Widget):
    score = NumericProperty(0)
    score_text = StringProperty("0")


class OverlapGauge(Widget):
    overlap = NumericProperty(0)

    def update_overlap(self, points: np.ndarray):
        window = (0, 0.6, 1.0, 1.0)
        overlap = rectangle_ops.points_window_overlap(points, window)
        print(overlap)
        anim = Animation(overlap=overlap, duration=0.5)
        anim.start(self)


class CameraCanvas(Widget):
    camera_widget: CameraWidget = ObjectProperty(None)
    matched_points = ListProperty([])
    group: Optional[InstructionGroup] = None
    line_points = ListProperty([(0, 0), (0, 0)])

    def update_overlay(self, score: float, points: np.ndarray, line: np.ndarray):

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
            color = Color(0 / 255, 133 / 255, 15 / 255, 0.8)
            self.group.add(color)
            rect = Ellipse(pos=(x, y), size=(24, 24))
            self.group.add(rect)

        self.canvas.add(self.group)

        line_points = []
        for point in line:
            x, y = point
            x, y = int(x * sx + ox), int(y * sy + oy)
            line_points.append((x, y))

        anim = Animation(line_points=line_points, duration=0.4)
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
    screen_pano: PreviewPanoramaWindow = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data = {}
        self.frame: np.ndarray = None
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.photo_index = 0
        self.last_save_image = None
        self.fe = cv2.SIFT_create()

        Clock.schedule_once(self.bind_pano_actions)

        Clock.schedule_interval(self.snapshot_frame, 0.25)

    def bind_pano_actions(self, *args):
        # https://stackoverflow.com/questions/48963808/binding-kivy-objectproperty-to-a-child-widget-doesnt-seem-to-work-outside-of-ro
        self.screen_pano.return_btn.bind(on_release=self.cancel_pano)
        self.screen_pano.accept_btn.bind(on_release=self.accept_pano)
        self.monitor_photo()

    @property
    def sm(self):
        return self.screen_manager

    def cancel_pano(self, *args):

        self.sm.current = "camera"
        self.camera.play = True

        if self.last_save_image is not None:
            if os.path.exists(self.last_save_image):
                os.remove(self.last_save_image)
                self.last_save_image = None

    def accept_pano(self, *args):

        self.data["photo"] = self.data["pano_proposal"]
        self.screen_pano.enable_accept(True)
        self.sm.current = "camera"
        self.camera.play = True

        image = self.data["photo"][2].copy()
        self.photo_index += 1
        storage.save_image(image, "stitched.jpg", self.session_id)

    def on_photo_captured(self, *args):
        photo = self.data["pano_proposal"][2]
        self.screen_pano.set_image(photo)
        self.screen_pano.enable_accept(True)
        self.sm.current = "pano"
        self.camera.play = False

    def cancel_photo(self, *args):
        self.screen_pano.enable_accept(True)
        self.data = {}
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.photo_index = 0

    def preview_pano(self, *args):
        if "photo" in self.data:
            self.data["pano_proposal"] = self.data["photo"]
            keypoints, des, image = self.data["photo"]
            image = image.copy()
            cv2.drawKeypoints(image, keypoints, image, color=(255, 0, 0))
            self.screen_pano.set_image(image)
            self.screen_pano.enable_accept(False)
            self.sm.current = "pano"
            self.camera.play = False

    def preview_matches(self, *args):
        if "matches_image" in self.data:
            self.take_photo(match_mode=True)
            matches_image = self.data["matches_image"]
            self.screen_pano.set_image(matches_image)
            self.screen_pano.enable_accept(False)
            self.sm.current = "pano"
            self.camera.play = False

    @property
    def camera(self) -> CameraWidget:
        return self.camera_canvas.camera_widget

    def snapshot_frame(self, *args):
        self.frame = self.camera.frame_from_buf()

    def analyse_photo(
        self, key: str, image: np.ndarray, resize: bool = True, mask=None
    ):

        if image is None:
            return False

        if resize:
            image = cv2.resize(image, CV_IMAGE_SIZE)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        print(f"Computing features ... {gray.shape}")

        keypoints, des = self.fe.detectAndCompute(gray, mask=mask)
        if mask is not None:
            indices = [
                i
                for i, k in enumerate(keypoints)
                if mask[int(k.pt[1]), int(k.pt[0])] > 0
            ]
            keypoints = [keypoints[i] for i in indices]
            des = des[indices]
        self.data[key] = (keypoints, des, image)
        print("Capturing image - done")
        return True

    def stitch(self, *args):

        if "photo" not in self.data:
            return

        if "next_photo" not in self.data:
            return

        kp1, des1, img1 = self.data["photo"]
        kp2, des2, img2 = self.data["next_photo"]

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        bf = cv2.FlannBasedMatcher(index_params, search_params)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) < 40:
            Snackbar(
                text=f"Not enough matches ({len(good)}) !",
                bg_color=(1, 0, 0, 0.5),
                duration=1,
            ).open()
            return False

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        print("STICZING!:", sum(matchesMask))

        good = [[p] for p, m in zip(good, matchesMask) if m == 1]
        matches_image = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, good, flags=2, outImg=None
        )
        self.data["matches_image"] = matches_image

        if sum(matchesMask) > 40:
            M_inv = np.linalg.inv(M)
            img2 = img2.copy()
            img1 = img1.copy()
            new_image, _ = transform.cv_blend_images(img2, img1, M_inv)
            mask, _ = transform.cv_blend_images(
                np.ones_like(img2), np.zeros_like(img1), M_inv
            )
            mask = mask[:, :, 0].astype(np.uint8) * 255

            cut_mask, _ = transform.cv_blend_images(
                np.ones_like(img2), np.ones_like(img1), M_inv
            )
            cut_mask = cut_mask[:, :, 0]

            # print(new_image.shape, mask.shape, cut_mask.shape)
            slice_mask = cut_mask[:, 0].copy()
            y_min = None
            y_max = None

            for i, m in enumerate(slice_mask):
                if y_min is None and m > 0:
                    y_min = i

                if y_max is None and m == 0 and y_min is not None:
                    y_max = i

            # x_min = 0
            # x_max = None
            #
            # for i in list(range(cut_mask.shape[1]))[::-1]:
            #     v = cut_mask[:, i].sum()
            #     if x_max is None and v >= slice_mask.sum():
            #         x_max = i
            #         break
            #
            # print(">> y_min: y_max, x_min: x_max", y_min, y_max, x_min, x_max, new_image.shape)

            x_min = 0
            x_max = cut_mask.shape[1] - 50

            new_image = new_image[y_min: y_max, x_min: x_max]
            mask = mask[y_min: y_max, x_min: x_max]

            self.analyse_photo("pano_proposal", new_image, resize=False, mask=mask)
            return True
        else:
            Snackbar(
                text=f"Not enough homography matches ({sum(matchesMask)}) !",
                bg_color=(1, 0, 0, 0.5),
                duration=2,
            ).open()
            return False

    def take_photo(self, match_mode=False, *args):

        image = self.camera.frame_from_buf()

        if not match_mode:
            name = f"{self.photo_index}".zfill(3) + ".jpg"
            self.last_save_image = storage.save_image(image, name, self.session_id)

        if "photo" not in self.data:
            self.photo_index += 1
            self.analyse_photo("photo", image)
        else:
            self.analyse_photo("next_photo", image)
            status = self.stitch()
            if status and not match_mode:
                self.on_photo_captured()

    def monitor_photo(self):
        threading.Thread(target=self.infinite_loop).start()

    def infinite_loop(self):
        iteration = 0
        while True:
            if not self.camera.play:
                time.sleep(0.5)
                continue
            if self.stop.is_set():
                # Stop running this thread so the main Python process can exit.
                return
            iteration += 1

            status = self.analyse_photo("current", self.frame)
            score, points, line = self.match_score()
            if not status or score is None:
                self.camera_canvas.update_overlay(0.0, [], line)
                time.sleep(0.5)
            else:
                print(f"Iteration {iteration} ")
                self.camera_canvas.update_overlay(score, points, line)
                time.sleep(0.5)

    def match_score(self):
        line = np.array([[0, 0], [0, CV_IMAGE_SIZE[1]]], dtype=np.float32)

        if "photo" not in self.data:
            return None, [], line

        if "current" not in self.data:
            return None, [], line

        current_pano_image = self.data["photo"][2]

        kp1, des1 = self.data["photo"][:2]
        kp2, des2 = self.data["current"][:2]
        H, matches = matching.match_images(kp1, des1, kp2, des2)
        print(f"Num matches: Nmatch={len(matches)}, Nkpi1={len(kp1)}, Nkpi2={len(kp2)}")

        matching_score = 200 * len(matches) / max(len(kp1), 1)

        _, current_photo_points = matching.select_matching_points(kp1, kp2, matches)
        normalized_points = current_photo_points / np.array([CV_IMAGE_SIZE])
        # inverting Y coordinates
        normalized_points[:, 1] = 1 - normalized_points[:, 1]

        height, width = current_pano_image.shape[:2]

        if H is not None:
            line = np.array([[width, 0], [width, height]], dtype=np.float32)
            line = transform.homography_transform(line, H)

        line = line / np.array([CV_IMAGE_SIZE])
        line[:, 1] = 1 - line[:, 1]

        line = np.clip(line, 0, 1)

        return matching_score, normalized_points, line


main_app_uix = """

<ContentNavigationDrawer>:

    ScrollView:

        MDList:

            OneLineListItem:
                text: "Settings"
                on_press:
                    root.nav_drawer.set_state("close")
                    root.screen_manager.current = "scr 1"
                    
            OneLineListItem:
                text: "Camera"
                on_press:
                    root.nav_drawer.set_state("close")
                    root.screen_manager.current = "camera"


<MainWindow>:
    camera_canvas: camera_canvas
    CameraCanvas:
        id: camera_canvas
        
    MDFloatingActionButton:
        text: "Capture"
        icon: "camera"
        pos_hint: {"center_x": 0.5, "center_y": 0.1}
        elevation: 8
        on_press: 
            self.parent.take_photo()
    
    MDFloatingActionButton:
        text: "Cancel"
        icon: "cancel"
        pos_hint: {"center_x": 0.8, "center_y": 0.1}
        elevation: 8
        on_press: 
            self.parent.cancel_photo()
            
    MDFloatingActionButton:
        text: "Preview"
        icon: "eye"
        pos_hint: {"center_x": 0.2, "center_y": 0.1}
        elevation: 8
        on_press: 
            self.parent.preview_pano()                

    MDFloatingActionButton:
        text: "Preview Matches"
        icon: "hexagon"
        pos_hint: {"center_x": 0.5, "center_y": 0.2}
        elevation: 8
        
        on_press: 
            self.parent.preview_matches()   


<MainApp>:
    nav_drawer: nav_drawer
    screen_manager: screen_manager
    camera_screen: camera_screen
    
    MDToolbar:
        id: toolbar
        pos_hint: {"top": 1}
        elevation: 10
        title: "Test"
        left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]

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
            
            PreviewPanoramaWindow:
                id: screen_pano
                name: "pano"
            
            Screen:
                name: "scr 1"

                MDLabel:
                    text: "Screen 1 test"
                    halign: "center"

                
        MDNavigationDrawer:
            id: nav_drawer

            ContentNavigationDrawer:
                screen_manager: screen_manager
                nav_drawer: nav_drawer
"""


class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()


class MainApp(Screen):
    camera_screen: MainWindow = ObjectProperty()


Builder.load_string(main_app_uix)


if kivy.platform == "linux":
    from kivy.core.window import Window

    Window.size = (480, 800)
    Window.top = 50
    Window.left = 1600


class MyApp(MDApp):
    def build(self):
        self.settings_cls = SettingsWithSidebar
        self.mainApp = MainApp()
        return self.mainApp

    def on_stop(self):
        print("Exiting!")
        self.mainApp.camera_screen.stop.set()
        super(MyApp, self).on_stop()


if __name__ == "__main__":
    MyApp().run()
