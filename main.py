import kivy
from kivy import Logger
from kivy.lang import Builder
from kivy.logger import LoggerHistory
from kivy.properties import ObjectProperty, ConfigParser, StringProperty, ListProperty
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.settings import Settings, SettingsWithTabbedPanel
from kivy.utils import escape_markup
from kivymd.app import MDApp
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import OneLineIconListItem, MDList
from kivymd.uix.navigationdrawer import MDNavigationDrawer

import settings as settings_ops
from uix.basic_stitcher import BasicStitcherScreen
from uix.camera_viewer_screen import CameraViewerScreen
from uix.keypoints_viewer_screen import KeypointsViewerScreen
from uix.left_to_right_stitcher import LeftToRightStitcherScreen
from uix.tracker_screen import TrackerScreen


Builder.load_file("main.kv")


class LoggerHistoryScreen(Screen):
    logs_label = ObjectProperty()

    def get_logger_history(self) -> str:
        lines = []
        for entry in LoggerHistory.history[::-1]:
            lines.append(entry.msg)
        text = "\n".join(lines)
        text = escape_markup(text)
        return f"[font=assets/fonts/VeraMono.ttf]{text}[/font]"

    def update_logger_history(self):
        self.logs_label.text = self.get_logger_history()

    def play(self):
        pass

    def pause(self):
        pass

class ItemDrawer(OneLineIconListItem):
    icon = StringProperty("eye")
    text_color = ListProperty((1, 1, 1, 1))


class DrawerList(ThemableBehavior, MDList):
    def set_color_item(self, instance_item):
        """Called when tap on a menu item."""

        # Set the color of the icon and text for the menu item.
        for item in self.children:
            if item.text_color == self.theme_cls.primary_color:
                item.text_color = self.theme_cls.text_color
                break
        instance_item.text_color = self.theme_cls.primary_color



class MainAppScreen(Screen):
    nav_drawer: MDNavigationDrawer = ObjectProperty()
    screen_manager: ScreenManager = ObjectProperty()
    logger_history: LoggerHistoryScreen = ObjectProperty()
    keypoints_viewer: KeypointsViewerScreen = ObjectProperty()
    camera_viewer: CameraViewerScreen = ObjectProperty()
    tracker_viewer: TrackerScreen = ObjectProperty()
    basic_stitcher_viewer: BasicStitcherScreen = ObjectProperty()
    left_to_right_stitcher_viewer: LeftToRightStitcherScreen = ObjectProperty()


if kivy.platform == "linux":
    from kivy.core.window import Window

    Window.size = (480, 800)
    Window.top = 50
    Window.left = 1600


if kivy.platform == "android":
    from android.permissions import request_permissions, Permission

    request_permissions(
        [
            Permission.CAMERA,
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
        ]
    )



class SticzingerApp(MDApp):
    running = True

    def build(self):

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        self.settings_cls = SettingsWithTabbedPanel
        self.use_kivy_settings = True

        self.main_screen = MainAppScreen()
        self.main_screen.nav_drawer.bind(
            state=self.nav_drawer_state_change
        )
        self.main_screen.screen_manager.current = "left-to-right-stitcher-viewer"
        return self.main_screen

    def on_stop(self):
        Logger.info("Exiting!")
        self.running = False
        super(SticzingerApp, self).on_stop()

    def close_nav_drawer(self):
        self.root.nav_drawer.set_state("close")

    def open_nav_drawer(self, instance):
        self.root.nav_drawer.set_state("open")

    def open_screen(self, item: ItemDrawer, name: str):
        if name != self.root.screen_manager.current:
            self.main_screen.keypoints_viewer.pause()
            self.main_screen.camera_viewer.pause()
            self.main_screen.tracker_viewer.pause()
            self.main_screen.basic_stitcher_viewer.pause()
            self.main_screen.left_to_right_stitcher_viewer.pause()
            # TODO maybe reset state for stitchers here ?

        self.close_nav_drawer()
        self.main_screen.ids.toolbar.title = item.text
        self.root.screen_manager.current = name

    def nav_drawer_state_change(self, instance, value):
        Logger.info(f"Navigation drawer state changed: {instance} {value}")

    def open_logger_history(self, item: ItemDrawer):
        self.main_screen.logger_history.update_logger_history()
        self.open_screen(item, self.main_screen.logger_history.name)

    def open_keypoints_viewer(self, item: ItemDrawer):
        self.open_screen(item, self.main_screen.keypoints_viewer.name)

    def open_camera_viewer(self, item: ItemDrawer):
        self.open_screen(item, self.main_screen.camera_viewer.name)

    def open_tracker_viewer(self, item: ItemDrawer):
        self.open_screen(item, self.main_screen.tracker_viewer.name)

    def open_basic_stitcher_viewer(self, item: ItemDrawer):
        self.open_screen(item, self.main_screen.basic_stitcher_viewer.name)

    def open_left_to_right_stitcher_viewer(self, item: ItemDrawer):
        self.open_screen(item, self.main_screen.left_to_right_stitcher_viewer.name)

    def open_app_settings(self, *args):
        self.main_screen.screen_manager.current_screen.pause()
        self.open_settings()

    def close_settings(self, *args):
        self.main_screen.screen_manager.current_screen.play()
        return super(SticzingerApp, self).close_settings(*args)

    def build_config(self, config: ConfigParser):
        settings_ops.AppSettings().build_config(config)

    def build_settings(self, settings: Settings):
        settings_ops.AppSettings().build_settings(self.config, settings)

    def on_config_change(self, config, section, key, value):
        Logger.info(f"Config changed: {section} {key} {value}")
        if key in ("keypoint_detector", "image_size"):
            self.main_screen.basic_stitcher_viewer.reset_state()
            self.main_screen.left_to_right_stitcher_viewer.reset_state()


if __name__ == "__main__":
    SticzingerApp().run()
