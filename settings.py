import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, List, Tuple, Dict

from kivy.config import ConfigParser
from kivy.uix.settings import Settings


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
            "ORB_FAST",
            "ORB_HARRIS",
            "ORB_FAST_512",
            "ORB_FAST_1024",
            "ORB_HARRIS_512",
            "ORB_HARRIS_1024",
            "FAST_SIFT",
            "FAST_SURF",
        ],
    )

    fast_threshold = OptionsProperty(
        10,
        "FAST threshold",
        "FAST keypoint detector threshold value",
        options=["10", "15", "20", "25", "30"],
    )

    fast_max_keypoints = OptionsProperty(
        1024,
        "FAST maximum keypoints",
        "FAST keypoint detector num keypoints to keep",
        options=["500", "1000", "1500", "2000"],
    )

    image_size = OptionsProperty(
        500,
        "Image Size",
        "Keypoints extraction image size",
        options=["250", "300", "350", "400", "500", "600"],
    )

    def get_image_size(self) -> Tuple[int, int]:
        size = self.image_size.value
        return size, size

    def get_keypoint_detector_kwargs(self):
        return {
            "name": self.keypoint_detector.value,
            "fast_max_keypoints": self.fast_max_keypoints.value,
            "fast_threshold": self.fast_threshold.value,
        }


class ImageMatchingConf(PropertiesGroup):
    name = TitleProperty("Image Matching Settings")
    cross_check = BoolProperty(
        "1", "MF Matcher Cross Check option", "Applicable only to BFMatcher"
    )
    matcher_type = OptionsProperty(
        "brute_force",
        "Matcher Type",
        "OpenCV matcher type",
        options=["brute_force", "flann", "tflite", "cython_brute_force"],
    )
    min_matches = OptionsProperty(
        40,
        "Keypoint Detector Min Matches",
        "minimum matches to accept photo",
        options=["20", "30", "40", "50"],
    )
    lowe_ratio = OptionsProperty(
        0.7,
        "Lowe's parameter",
        "Applicable only when Matcher is Flann or cross_check is set to False",
        options=["0.5", "0.6", "0.7", "0.8", "0.9"],
    )
    ransack_threshold = OptionsProperty(
        7,
        "RANSACK threshold",
        "Reprojection error threshold",
        options=["5", "7", "10", "14"],
    )


class AppSettings(PropertiesGroup):
    stitching_conf = StitchingConf()
    keypoints_extractor_conf = KeypointsExtractorConf()
    matching_conf = ImageMatchingConf()

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

    @property
    def matching_configuration(self) -> Dict[str, Any]:

        cross_check = self.matching_conf.cross_check.value
        detector_type = self.keypoints_extractor_conf.keypoint_detector.value.upper()

        if "SIFT" in detector_type or "KAZE" in detector_type or "SURF" in detector_type:
            bf_matcher_norm = "NORM_L2"
        else:
            bf_matcher_norm = "NORM_HAMMING"

        conf = {
            "bf_matcher_cross_check": cross_check,
            "bf_matcher_norm": bf_matcher_norm,
            "lowe": self.matching_conf.lowe_ratio.value,
            "ransack_threshold": self.matching_conf.ransack_threshold.value,
            "matcher_type": self.matching_conf.matcher_type.value,
        }
        return conf
