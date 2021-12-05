import numpy as np
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivymd.uix.button import MDFloatingActionButton

import cameras
from logging_ops import profile

SCATTER_UIX = """
# kv_start

<CustomScrollView@ScrollView>:
    do_scroll_x: False
    do_scroll_y: False

    canvas.before:
        Color:
            rgba: 0.5, 0.5, 0.5, 0.5
        Rectangle:
            size: self.size
            pos: self.pos
# kv_end
"""

Builder.load_string(SCATTER_UIX)



class CustomScrollView(ScrollView):
    pass


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

        scroll_view = CustomScrollView()
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
        cameras.copy_image_to_texture(image, self.render)

    def enable_accept(self, toggle: bool = True):
        if toggle and hasattr(self.accept_btn, "saved_pos_hint"):
            self.accept_btn.pos_hint = self.accept_btn.saved_pos_hint
            return
        if not toggle:
            self.accept_btn.saved_pos_hint = self.accept_btn.pos_hint
            self.accept_btn.pos_hint = {"center_x": 0.0, "center_y": -0.1}

    @profile()
    def cancel(self, *args):
        self.cancel_callback(self.manager)

    @profile()
    def accept(self, *args):
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
