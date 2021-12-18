from typing import Optional, Tuple

import cv2
import numpy as np


class CVExtractor:
    def extract(self, image, mask: Optional[np.ndarray] = None):
       pass


class CVWrapper(CVExtractor):
    def __init__(self, fe):
        self.fe = fe

    def extract(self, image, mask: Optional[np.ndarray] = None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.fe.detectAndCompute(gray, mask=mask)


class FASTSIFT(CVExtractor):
    def __init__(self, threshold: int = 10, max_features: int = 1024):
        self.fast = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        )
        self.fe = cv2.SIFT_create(nfeatures=max_features)
        self.max_features = max_features

    def extract(self, image, mask: Optional[np.ndarray] = None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints = self.fast.detect(gray, mask)

        keypoints = sorted(keypoints, key=lambda p: p.response, reverse=True)
        keypoints = keypoints[:self.max_features]
        keypoints, descriptors = self.fe.compute(gray, keypoints)
        return keypoints, descriptors


class FASTSURF(FASTSIFT):
    def __init__(self, threshold: int = 10, max_features: int = 1024):
        self.fast = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        )
        self.fe = cv2.xfeatures2d.SURF_create()
        self.max_features = max_features


def extract_keypoints(
    image,
    dsize: Optional[Tuple[int, int]],
    extractor: CVWrapper,
    mask: Optional[np.ndarray] = None,
):
    if dsize is not None:
        image = cv2.resize(image, dsize=dsize)
    keypoints, des = extractor.extract(image, mask)
    return keypoints, des


def create_keypoint_extractor(name: str, fast_max_keypoints: int, fast_threshold: int) -> CVExtractor:
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
    elif value == "FAST_SIFT":
        return FASTSIFT(fast_threshold, fast_max_keypoints)
    elif value == "FAST_SURF":
        return FASTSURF(fast_threshold, fast_max_keypoints)
    else:
        raise NotImplementedError(value)

    return CVWrapper(fe)


def detect_keypoints(
    image: Optional[np.ndarray], dsize: Tuple[int, int], extractor: CVWrapper
) -> np.ndarray:

    if image is None:
        return np.zeros([0, 2])

    keypoints, _ = extract_keypoints(image, dsize, extractor)
    points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]

    normalized_points = points / np.array([dsize])
    return normalized_points
