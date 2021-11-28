import time

import cv2
import kivy
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty

# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image

# from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.scrollview import ScrollView

# from kivy.uix.slider import Slider
from kivymd.uix.card import MDCard
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.screen import MDScreen as Screen, MDScreen
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel as Label, MDLabel
from kivymd.uix.slider import MDSlider as Slider
from kivymd.uix.button import MDFlatButton as Button, MDRectangleFlatButton
from kivymd.uix.boxlayout import MDBoxLayout as BoxLayout, MDBoxLayout

import transform


class AndroidCamera(Camera):
    camera_resolution = (640 * 2, 480 * 2)
    counter = 0

    def _camera_loaded(self, *largs):
        self.texture = Texture.create(
            size=np.flip(self.camera_resolution), colorfmt="rgb"
        )
        self.texture_size = list(self.texture.size)

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None
        frame = self.frame_from_buf()

        self.frame_to_screen(frame)
        super(AndroidCamera, self).on_tex(*l)

    def frame_from_buf(self):
        buffer = self._camera._buffer
        if buffer is None:
            return None

        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), "uint8").reshape(
            (h + h // 2, w)
        )
        frame_bgr = cv2.cvtColor(frame, 93)
        return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(
            frame_rgb,
            str(self.counter),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self.counter += 1
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tostring()
        self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")


class LinuxCamera(Camera):
    camera_resolution = (-1, -1)
    counter = 0

    def frame_from_buf(self):
        image = np.frombuffer(self.texture.pixels, np.uint8).reshape(
            *self.texture.size[::-1], 4
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class CameraWidget(Camera):
    index = 0
    allow_stretch = True
    play = True

    def __new__(cls, *args, **kwargs):
        if kivy.platform != "linux":
            return AndroidCamera(
                *args, **kwargs, resolution=AndroidCamera.camera_resolution
            )
        else:
            return LinuxCamera(
                *args, **kwargs, resolution=LinuxCamera.camera_resolution
            )


def numpy_to_image(img, image: Image):
    h, w, _ = img.shape
    # texture = Texture.create(size=(w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.flip(img, 0)
    image.texture = Texture.create(size=(w, h), colorfmt="rgb")
    image.texture.blit_buffer(img.flatten(), colorfmt="rgb", bufferfmt="ubyte")
    # return texture
    # return Image(size=(w, h), texture=texture)


class MyLayout(BoxLayout):
    orientation = "vertical"
    slider_val = NumericProperty(100)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.data = {}
        layout = self
        self.camera = CameraWidget(allow_stretch=True)
        self.slider = Slider(min=50, max=500, value=100, hint_radius=20)
        self.slider.fbind("value", self.on_slider_val)
        self.label = Label(text=str(self.slider_val), size_hint=(0.2, 1.0))

        self.pictureA = Image(source="assets/template.jpg")
        self.pictureB = Image(source="assets/templateB.jpg")

        buttonA = MDRectangleFlatButton(text="Photo A", size_hint=(1, 1.0))
        buttonB = MDRectangleFlatButton(text="Photo B", size_hint=(1, 1.0))
        buttonMatch = MDRectangleFlatButton(text="Match", size_hint=(1, 1.0))

        layout.add_widget(self.camera)

        hlayout = MDBoxLayout(pos_hint={"center_x": 0.5}, size_hint=(1, 0.25))
        hlayout.add_widget(self.pictureA)
        hlayout.add_widget(self.pictureB)

        layout.add_widget(hlayout)

        hlayout = BoxLayout(orientation="horizontal", size_hint=(1.0, 0.15))
        hlayout.add_widget(self.slider)
        layout.add_widget(hlayout)

        hlayout = MDBoxLayout(size_hint=(1.0, 0.15), spacing=5, padding=5)
        hlayout.add_widget(buttonA)
        hlayout.add_widget(buttonB)
        hlayout.add_widget(buttonMatch)
        layout.add_widget(hlayout)

        buttonA.bind(on_press=self.take_photo_A)
        buttonB.bind(on_press=self.take_photo_B)
        buttonMatch.bind(on_press=self.match)

    def on_slider_val(self, instance, val):
        self.label.text = str(int(val))

    def analyse_photo(self, key, picture):

        image = self.camera.frame_from_buf()

        orb = cv2.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # gray = cv2.resize(gray, (512, 512))
        keypoint, des = orb.detectAndCompute(gray, None)
        img_final = cv2.drawKeypoints(
            image, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        self.data[key] = (keypoint, des, image)
        numpy_to_image(img_final, picture)

    def take_photo_A(self, *args):
        self.analyse_photo("A", self.pictureA)

    def take_photo_B(self, *args):
        self.analyse_photo("B", self.pictureB)

    def match(self, *args):

        # BFMatcher with default params
        if "A" not in self.data:
            return
        if "B" not in self.data:
            return

        kp1, des1, img1 = self.data["A"]
        kp2, des2, img2 = self.data["B"]

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

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if sum(matchesMask) > self.slider.value:

            h, w, _ = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(pts, M)
            print("Homography", M)
            print("mask", sum(mask))

            print("img2", img2.shape, img2.dtype)
            print("dst", dst)
            color = (255, 0, 0)
            points = np.int32(dst).reshape([-1, 1, 2])

            # img3 = cv2.polylines(img2, pts=[points], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

            img3, _ = transform.cv_blend_images(img2, img1, np.linalg.inv(M))

        else:
            good = [[p] for p, m in zip(good, matchesMask) if m == 1]

            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2, outImg=None)

        render = Image()
        numpy_to_image(img3, render)

        root = ScrollView(do_scroll_x=False, do_scroll_y=False)

        scatter = Scatter(do_rotation=False)
        # dismiss = Button(text="close", on_press = self.popup.dismiss())
        # image = Image(source='sun.jpg')
        scatter.add_widget(render)
        scatter.scale = 10
        root.add_widget(scatter)

        def play(*args):
            print("Resuming")
            self.camera.play = True

        popup = Popup(
            title=f"Num matches: {sum(matchesMask)}", content=root, size_hint=(1, 0.8),
            on_dismiss = play
        )
        self.camera.play = False

        popup.open()
        popup.on_dismiss()


class MenuScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        layout = MyLayout()
        self.add_widget(layout)


class TrackerWidget(BoxLayout):
    orientation = "vertical"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = self
        self.camera = CameraWidget(allow_stretch=True)
        self.image = Image(source="assets/template.jpg")
        layout.add_widget(self.image)

        nn_thresh = 0.7
        max_length = 15
        self.nn_thresh = nn_thresh
        self.min_length = 3
        import tracker

        # This class helps merge consecutive point matches into tracks.
        self.tracker = tracker.PointTracker(max_length, nn_thresh=nn_thresh)

        self.fe = cv2.SIFT_create(
            edgeThreshold=2, contrastThreshold=0.1, nOctaveLayers=2, sigma=0.5
        )
        # self.fe = cv2.ORB_create(1000, scoreType=cv2.ORB_FAST_SCORE)

        self.edgeThreshold = Slider(min=10, max=100, value=20, hint_radius=20)
        self.edgeThreshold.fbind("value", self.on_slider_val)

        self.contrastThreshold = Slider(min=5, max=100, value=10, hint_radius=20)
        self.contrastThreshold.fbind("value", self.on_slider_val)

        self.nOctaveLayers = Slider(min=1, max=8, value=2, hint_radius=20)
        self.nOctaveLayers.fbind("value", self.on_slider_val)

        self.sigma = Slider(min=10, max=200, value=50, hint_radius=20)
        self.sigma.fbind("value", self.on_slider_val)

        self.on_slider_val(None, None)

        slayout = MDBoxLayout(orientation="vertical", size_hint=(1, 0.2), padding=20)
        slayout.add_widget(self.edgeThreshold)
        slayout.add_widget(self.contrastThreshold)
        slayout.add_widget(self.nOctaveLayers)
        slayout.add_widget(self.sigma)
        layout.add_widget(slayout)

        Clock.schedule_interval(self.track_movement, 0.05)

    def on_slider_val(self, instance, val):
        print(
            f"sigma = {self.sigma.value / 100.0} "
            f"nOctaveLayers={int(self.nOctaveLayers.value)} "
            f"contrastThreshold={self.contrastThreshold.value / 1000} "
            f"edgeThreshold={self.edgeThreshold.value / 10}"
        )
        self.fe = cv2.SIFT_create(
            edgeThreshold=self.edgeThreshold.value / 10,
            contrastThreshold=self.contrastThreshold.value / 1000,
            nOctaveLayers=int(self.nOctaveLayers.value),
            sigma=self.sigma.value / 100.0,
        )

    def track_movement(self, dt):

        image = self.camera.frame_from_buf()
        if image is None:
            return

        image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        start1 = time.time()
        pts, desc = self.fe.detectAndCompute(gray, None)

        if desc is None:
            return None


        pts = np.array([[*p.pt, 0.0] for p in pts])
        pts = pts.T
        desc = desc.T.astype(np.float32)

        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        desc = np.concatenate([pts, desc], axis=0)

        self.tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = self.tracker.get_tracks(self.min_length)

        # Font parameters for visualizaton.
        font = cv2.FONT_HERSHEY_DUPLEX
        font_clr = (255, 255, 255)
        font_pt = (4, 20)
        font_sc = 0.8

        end1 = time.time()
        delta = end1 - start1

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((gray, gray, gray))).astype('uint8')
        tracks[:, 1] /= float(self.nn_thresh)  # Normalize track scores to [0,1].
        self.tracker.draw_tracks(out1, tracks)

        out1 = cv2.resize(out1, self.camera.texture_size)
        cv2.putText(out1, f'Point Tracks - {delta*1000:.2f} [ms]', font_pt, font, font_sc, font_clr, lineType=16)

        # out2 = (np.dstack((gray, gray, gray))).astype("uint8")
        # for pt in pts.T:
        #     pt1 = (int(round(pt[0])), int(round(pt[1])))
        #     cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        #
        # out2 = cv2.resize(out2, self.camera.texture_size)
        # cv2.putText(
        #     out2, f"Raw Point Detections - {delta*1000:.2f} [ms]", font_pt, font, font_sc, font_clr, lineType=16
        # )

        numpy_to_image(out1, self.image)


class MenuScreen2(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        layout = TrackerWidget()
        self.add_widget(layout)


if kivy.platform == "linux":
    from kivy.core.window import Window

    Window.size = (480, 800)


class MyApp(MDApp):
    def build(self):

        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu1'))
        # sm.add_widget(MenuScreen2(name="menu2"))

        return sm


if __name__ == "__main__":
    MyApp().run()
