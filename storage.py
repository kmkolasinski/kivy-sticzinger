import json
from pathlib import Path
from typing import Dict, Any

import cv2
from kivy.utils import platform

from logging_ops import profile


def get_save_path() -> Path:
    if platform == "android":
        from android.permissions import request_permissions, Permission

        request_permissions(
            [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        )
        PATH = "/storage/emulated/0/DCIM/Sticzinger/images"
    else:
        PATH = "data/DCIM/images"

    Path(PATH).mkdir(exist_ok=True, parents=True)

    return Path(PATH)


@profile(name="storage.save_image")
def save_image(image, name: str, session: str = None) -> Path:

    PATH = get_save_path() / session
    Path(PATH).mkdir(exist_ok=True, parents=True)

    nomedia_file = PATH / ".nomedia"
    if not nomedia_file.exists():
        nomedia_file.open("w").close()

    save_path = PATH / f"{name}"

    print(f"Saving image {image.shape} to: {save_path}")

    cv2.imwrite(str(save_path), image)

    return save_path


def save_json(data: Dict[str, Any], filepath: str) -> Path:
    PATH = get_save_path()

    save_path = PATH / f"{filepath}"

    print(f"Saving JSON data to: {save_path}")

    with save_path.open("w") as file:
        json.dump(data, file, indent=2)

    return save_path