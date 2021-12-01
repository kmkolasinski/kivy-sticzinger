import json
from typing import Any, Union, List, Tuple

import cv2
from kivy.config import ConfigParser
from kivy.uix.settings import Settings
import dataclasses
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class BaseProperty:
    @property
    def type(self) -> str:
        raise NotImplementedError

    @property
    def asdict(self) -> dict:
        config = dataclasses.asdict(self)
        config["type"] = self.type
        return config

    def to_kivy_property_dict(self, section: str, key: str) -> dict:
        config = self.asdict
        if "default" in config:
            del config["default"]
        config["section"] = section
        config["key"] = key
        return config


@dataclass
class Property(BaseProperty):
    default: Union[str, int, float]
    title: str
    desc: str

    @property
    def config(self) -> ConfigParser:
        return ConfigParser.get_configparser("app")

    def _get_config_value(self) -> str:
        return self.config.get(self.section_injection, self.key_injection)

    @property
    def dtype(self) -> type:
        return type(self.default)

    @property
    def value(self) -> Union[str, int, float]:
        return self.dtype(self._get_config_value())


@dataclass
class TitleProperty(BaseProperty):
    title: str

    @property
    def type(self) -> str:
        return "title"

    def to_kivy_property_dict(self, section: str, key: str) -> dict:
        return {"title": self.title, "type": self.type}


@dataclass
class BoolProperty(Property):
    @property
    def type(self) -> str:
        return "bool"

    @property
    def value(self) -> Any:
        value = self._get_config_value()
        return value == "1"


@dataclass
class OptionsProperty(Property):

    options: List[str]

    @property
    def type(self) -> str:
        return "options"


class PropertiesGroup:
    name: TitleProperty = None

    @property
    def section_name(self) -> str:
        return self.name.title

    def __getattribute__(self, item: str):
        obj = getattr(type(self), item)
        if isinstance(obj, Property):
            obj.section_injection = self.section_name
            obj.key_injection = item
            return obj
        return super(PropertiesGroup, self).__getattribute__(item)

    def is_property(self, item: str):
        return isinstance(getattr(type(self), item), BaseProperty)

    def properties(self) -> List[Tuple[str, str, BaseProperty]]:
        properties = []
        for item in self.__dir__():
            obj = getattr(type(self), item)
            if isinstance(obj, PropertiesGroup):
                properties += obj.properties()
            elif isinstance(obj, BaseProperty):
                properties.append((self.section_name, item, obj))
        return properties


class StitchingConf(PropertiesGroup):
    name = TitleProperty("Stitching Settings")
    auto_next_image = BoolProperty(
        "1", "Auto next image", "Skip panorama preview after taking photo"
    )
    thread_max_wait_time = OptionsProperty(
        0.2,
        "Processing thread max wait time",
        "",
        options=["0.05", "0.1", "0.2", "0.3", "0.4", "0.5"],
    )


class KeypointsExtractorConf(PropertiesGroup):
    name = TitleProperty("Keypoints Extractor Settings")
    keypoint_detector = OptionsProperty(
        "SIFT",
        "Detector",
        "OpenCV keypoint detector type. Affects the speed of application",
        options=[
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
    )

    image_size = OptionsProperty(
        500,
        "Image Size",
        "Keypoints extraction image size",
        options=["300", "350", "400", "500", "600"],
    )

    def get_image_size(self) -> Tuple[int, int]:
        size = self.image_size.value
        return size, size


class AppSettings(PropertiesGroup):
    stitching_conf = StitchingConf()
    keypoints_extractor_conf = KeypointsExtractorConf()

    def get_kivy_settings(self):
        settings = []
        for section, key, item in self.properties():
            settings.append(item.to_kivy_property_dict(section, key))
        return settings

    def build_config(self, config: ConfigParser):

        settings = []
        for section, key, item in self.properties():
            settings.append((section, key, item.asdict))

        defaults = defaultdict(dict)
        for section, key, item in settings:
            if "default" in item:
                defaults[section][key] = item["default"]
                del item["default"]

        for section, values in defaults.items():
            config.setdefaults(section, values)

    def build_settings(self, config: ConfigParser, settings: Settings):
        settings.add_json_panel(
            "Configuration", config, data=json.dumps(self.get_kivy_settings())
        )

    # def get_keypoint_detector(self):
    #     value = self.keypoints_extractor_conf.keypoint_detector
    #     if value == "SIFT":
    #         return cv2.SIFT_create()
    #     elif value == "BRISK":
    #         return cv2.BRISK_create()
    #     elif value == "KAZE":
    #         return cv2.KAZE_create(extended=False)
    #     elif value == "KAZE_EXTENDED":
    #         return cv2.KAZE_create(extended=True)
    #     elif value == "ORB_FAST":
    #         return cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
    #     elif value == "ORB_HARRIS":
    #         return cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
    #     elif value == "ORB_FAST_512":
    #         return cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_FAST_SCORE)
    #     elif value == "ORB_FAST_1024":
    #         return cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_FAST_SCORE)
    #     elif value == "ORB_HARRIS_512":
    #         return cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_HARRIS_SCORE)
    #     elif value == "ORB_HARRIS_1024":
    #         return cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_HARRIS_SCORE)
    #     else:
    #         raise NotImplementedError(value)


# APP_SETTINGS = [
#     {"type": "title", "title": "Stitching Settings"},
#     {
#         "key": "auto_next_image",
#         "type": "bool",
#         "title": "Auto next image",
#         "desc": "Skip panorama preview after taking photo",
#         "section": STITCHING_SEC_KEY,
#         "default": "1",
#     },
#     # Keypoints Extractor Settings
#     {"type": "title", "title": "Keypoints Extractor Settings"},
#     {
#         "key": "keypoint_detector",
#         "type": "options",
#         "title": "Keypoint Detector",
#         "desc": "OpenCV keypoint detector type. Affects the speed of application",
#         "options": [
#             "SIFT",
#             "BRISK",
#             "KAZE",
#             "KAZE_EXTENDED",
#             "ORB_FAST",
#             "ORB_HARRIS",
#             "ORB_FAST_512",
#             "ORB_FAST_1024",
#             "ORB_HARRIS_512",
#             "ORB_HARRIS_1024",
#         ],
#         "section": KEYPOINTS_SEC_KEY,
#         "default": "SIFT",
#     },
#     {
#         "key": "image_size",
#         "type": "options",
#         "title": "Image Size",
#         "desc": "Image size used to stitch images, the lower the faster.",
#         "options": ["300", "350", "400", "500", "600"],
#         "section": KEYPOINTS_SEC_KEY,
#         "default": "500",
#     },
#     # Matcher Settings
#     {"type": "title", "title": "Matcher Settings"},
#     {
#         "key": "cross_check",
#         "type": "bool",
#         "title": "MF Matcher Cross Check option",
#         "desc": "Applicable only to BFMatcher",
#         "section": "Matcher Settings",
#         "default": "1",
#     },
#     {
#         "key": "matcher_type",
#         "type": "options",
#         "title": "Matcher Type",
#         "desc": "OpenCV matcher type",
#         "options": ["brute_force", "flann"],
#         "section": "Matcher Settings",
#         "default": "brute_force",
#     },
#     {
#         "key": "min_matches",
#         "type": "options",
#         "title": "Keypoint Detector Min Matches",
#         "desc": "minimum matches to accept photo",
#         "options": ["20", "30", "40", "50"],
#         "section": "Matcher Settings",
#         "default": "40",
#     },
#     {
#         "key": "lowe_ratio",
#         "type": "options",
#         "title": "Lowe's parameter",
#         "desc": "Applicable only when Matcher is Flann or cross_check is set to False",
#         "options": ["0.5", "0.6", "0.7", "0.8", "0.9"],
#         "section": "Matcher Settings",
#         "default": "0.7",
#     },
#     {
#         "key": "ransack_threshold",
#         "type": "options",
#         "title": "RANSACK threshold",
#         "desc": "Reprojection error threshold",
#         "options": ["5", "7", "10", "14"],
#         "section": "Matcher Settings",
#         "default": "7",
#     },
# ]


# class AppSettings:
#     @property
#     def config(self) -> ConfigParser:
#         return ConfigParser.get_configparser("app")
#
#     @property
#     def auto_next_image(self) -> bool:
#         value = self.config.get("Settings", "auto_next_image")
#         return value == "1"
#
#     @property
#     def resize_size(self) -> tuple[int, int]:
#         value = self.config.get("Settings", "preview_matching_size")
#         value = int(value)
#         return value, value
#
#     @property
#     def fe(self):
#         value = self.config.get("Settings", "keypoint_detector").upper()
#         if value == "SIFT":
#             return cv2.SIFT_create()
#         elif value == "BRISK":
#             return cv2.BRISK_create()
#         elif value == "KAZE":
#             return cv2.KAZE_create(extended=False)
#         elif value == "KAZE_EXTENDED":
#             return cv2.KAZE_create(extended=True)
#         elif value == "ORB_FAST":
#             return cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
#         elif value == "ORB_HARRIS":
#             return cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
#         elif value == "ORB_FAST_512":
#             return cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_FAST_SCORE)
#         elif value == "ORB_FAST_1024":
#             return cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_FAST_SCORE)
#         elif value == "ORB_HARRIS_512":
#             return cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_HARRIS_SCORE)
#         elif value == "ORB_HARRIS_1024":
#             return cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_HARRIS_SCORE)
#         else:
#             raise NotImplementedError(value)
#
#     @property
#     def min_matches(self) -> int:
#         value = self.config.get("Settings", "min_matches")
#         return int(value)
#
#     @property
#     def matching_configuration(self) -> dict[str, Any]:
#         key = "Matcher Settings"
#         cross_check = self.config.get(key, "cross_check") == "1"
#
#         detector_type = self.config.get("Settings", "keypoint_detector").upper()
#
#         if "SIFT" in detector_type or "KAZE" in detector_type:
#             bf_matcher_norm = "NORM_L2"
#         else:
#             bf_matcher_norm = "NORM_HAMMING"
#
#         conf = {
#             "bf_matcher_cross_check": cross_check,
#             "bf_matcher_norm": bf_matcher_norm,
#             "lowe": float(self.config.get(key, "lowe_ratio")),
#             "ransack_threshold": float(self.config.get(key, "ransack_threshold")),
#             "matcher_type": self.config.get(key, "matcher_type"),
#         }
#         return conf