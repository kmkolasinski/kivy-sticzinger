from typing import Optional, Tuple

import cv2
import numpy as np


class CVWrapper:
    def __init__(self, fe):
        self.fe = fe

    def extract(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.fe.detectAndCompute(gray, mask=None)


def extract_keypoints(image, dsize: Tuple[int, int], extractor: CVWrapper):
    image = cv2.resize(image, dsize=dsize)
    keypoints, des = extractor.extract(image)
    return keypoints, des


def create_keypoint_extractor(name: str) -> CVWrapper:
    value = name.upper()

    if value == "SIFT":
        fe = cv2.SIFT_create()
    elif value == "BRISK":
        fe = cv2.BRISK_create()
    elif value == "ORB_FAST":
        fe = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
    elif value == "ORB_HARRIS":
        fe = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
    elif value == "ORB_FAST_512":
        fe = cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_FAST_SCORE)
    elif value == "ORB_FAST_1024":
        fe = cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_FAST_SCORE)
    elif value == "ORB_HARRIS_512":
        fe = cv2.ORB_create(nfeatures=512, scoreType=cv2.ORB_HARRIS_SCORE)
    elif value == "ORB_HARRIS_1024":
        fe = cv2.ORB_create(nfeatures=1024, scoreType=cv2.ORB_HARRIS_SCORE)
    elif value == "KAZE":
        fe = cv2.KAZE_create()
    elif value == "AKAZE":
        fe = cv2.AKAZE_create()
    else:
        raise NotImplementedError(value)

    return CVWrapper(fe)


def detect_screen_keypoints(
    image: Optional[np.ndarray], dsize: Tuple[int, int], extractor: CVWrapper
) -> np.ndarray:

    if image is None:
        return np.zeros([0, 2])

    keypoints, _ = extract_keypoints(image, dsize, extractor)
    points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    normalized_points = points / np.array([dsize])

    # inverting Y coordinates
    normalized_points[:, 1] = 1 - normalized_points[:, 1]
    return normalized_points
