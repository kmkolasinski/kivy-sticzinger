import unittest
import time
from sticzinger_ops import *
import numpy as np
import cv2


class measuretime:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.seconds = time.perf_counter() - self.t
        print(f"{self.name}: took {self.seconds:6.4f} [s]")


def benchmark(name: str, method, steps=100, warmup=10):
    print()
    with measuretime(f"{name} warmup"):
        for _ in range(warmup):
            method()

    with measuretime(f"{name} calls "):
        for _ in range(steps):
            method()


class TestConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.random.randint(0, 128, (512, 256), dtype=np.uint8)

    def test_uint8_array2d_to_ascii_v1(self):
        benchmark("uint8_array2d_to_ascii_v1", lambda: uint8_array2d_to_ascii_v1(self.X))

    def test_uint8_array2d_to_ascii_v2(self):
        benchmark("uint8_array2d_to_ascii_v2", lambda: uint8_array2d_to_ascii_v2(self.X))

    def test_uint8_array2d_to_ascii(self):
        benchmark("uint8_array2d_to_ascii", lambda: uint8_array2d_to_ascii(self.X))

    def test_compare_values(self):

        v1 = uint8_array2d_to_ascii_v1(self.X)
        v2 = uint8_array2d_to_ascii_v2(self.X)
        v3 = uint8_array2d_to_ascii(self.X)
        self.assertEqual(v1, v2)
        self.assertEqual(v1, v3)
        self.assertEqual(len(v3), 512 * 256)

        X = np.array([ord(x) for x in v3]).reshape(512, 256)
        np.testing.assert_equal(X, self.X)

    def test_post_process(self):
        data = np.random.randint(0, 256, (512, 2)).astype(np.float32)
        kp1 = [create_kp(*d) for d in data]
        kp2 = [create_kp(*d) for d in data]

        data = np.array([[i, i, 1] for i in range(data.shape[0])])
        data = data.astype(np.float32)

        outputs = postprocess_and_refine_predictions(data, kp1, kp2)
        outputs

        py_outputs = py_postprocess(data, kp1, kp2)
        py_outputs


def create_kp(x, y):
    params = dict(
        x=x,
        y=y,
        _size=1,
        _angle=0,
        _response=1,
        _octave=0,
        _class_id=0,
    )
    return cv2.KeyPoint(**params)


def py_postprocess(
        predictions,
        kp1,
        kp2,
        threshold = 10.0,
        confidence = 0.99,
        homography_refine = True
):

    data = predictions.astype(np.int32)
    matches = data[data[:, 2] == 1][:, :2]

    cv_matches = [
        cv2.DMatch(_imgIdx=0, _queryIdx=q, _trainIdx=t, _distance=0)
        for q, t in matches
    ]

    H = None

    if homography_refine:
        src_pts = np.float32([kp1[m[0]].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m[1]].pt for m in matches]).reshape(
            -1, 1, 2
        )
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=threshold, confidence=confidence
        )

        cv_matches = [
            match
            for match, mask_val in zip(cv_matches, mask.ravel())
            if mask_val == 1
        ]

    return H, cv_matches