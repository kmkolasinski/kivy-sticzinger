from pathlib import Path

from PIL import Image
from kivy.utils import platform
import cv2

def save_image(image, name: str, session: str = None):
    if platform == "android":
        from android.permissions import request_permissions, Permission

        request_permissions(
            [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        )
        PATH = "/storage/emulated/0/DCIM/Sticzinger/"
        Path(PATH).mkdir(exist_ok=True)
    else:
        PATH = "data/DCIM"

    PATH = Path(PATH) / "images" / session

    Path(PATH).mkdir(exist_ok=True, parents=True)


    nomedia_file = PATH / ".nomedia"
    if not nomedia_file.exists():
        nomedia_file.open("w").close()

    save_path = PATH / f"{name}"

    print(f"Saving image {image.shape} to: {save_path}")
    # Image.fromarray(image).save(str(save_path))
    cv2.imwrite(str(save_path), image)

    return save_path
