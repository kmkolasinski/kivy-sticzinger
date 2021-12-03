import numpy as np
from kivy import Logger
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivymd.uix.button import MDFloatingActionButton

import cameras


class PreviewPanoramaScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parent_screen = None
        self.accept_callback = lambda: None
        self.cancel_callback = lambda: None

        self.cancel_btn = MDFloatingActionButton(
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

        self.add_widget(scroll_view)
        self.add_widget(self.cancel_btn)
        self.add_widget(self.accept_btn)

        self.cancel_btn.bind(on_release=self.cancel)
        self.accept_btn.bind(on_release=self.accept)

    def set_image(self, image):
        cameras.numpy_to_image(image, self.render)

    def enable_accept(self, toggle: bool = True):
        if toggle and hasattr(self.accept_btn, "saved_pos_hint"):
            self.accept_btn.pos_hint = self.accept_btn.saved_pos_hint
            return
        if not toggle:
            self.accept_btn.saved_pos_hint = self.accept_btn.pos_hint
            self.accept_btn.pos_hint = {"center_x": 0.0, "center_y": -0.1}

    def cancel(self, *args):
        Logger.info(f"Calling Cancel: {args}")
        self.cancel_callback(self.manager)

    def accept(self, *args):
        Logger.info(f"Calling Accept: {args}")
        self.accept_callback(self.manager)

    def show(
        self,
        manager: ScreenManager,
        image: np.ndarray,
        accept_callback=lambda: None,
        cancel_callback=lambda: None,
    ):
        self.set_image(image)
        self.accept_callback = accept_callback
        self.cancel_callback = cancel_callback

        self.enable_accept(self.accept_callback is not None)

        manager.switch_to(self)
