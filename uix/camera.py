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


camera_canvas = """
<CameraLayout>:   
    canvas.before:
        Color:
            rgba: 0/255, 0/255, 0/255, 0.1
        Rectangle:
            pos: self.pos
            size: self.size    

    CameraWidget:
        id: camera_widget
        allow_stretch: True
        pos: self.parent.pos
        size: self.parent.size
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