import unittest

import cv2
import numpy as np
from sticzinger_ops import *
from sticzinger_utils import measuretime


def benchmark(name: str, method, steps=500, warmup=10):
    print()
    with measuretime(f"{name} warmup", num_steps=steps):
        for _ in range(warmup):
            method()

    with measuretime(f"{name} calls ", num_steps=steps):
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



    def test_run_blas_dot(self):

        A = np.random.randn(1005, 128).astype(np.float32)
        B = np.random.randn(1000, 128).astype(np.float32)
        C = np.random.randn(1005, 1000).astype(np.float32)

        euclidean_dist_matrix(A, B, C)

        error = np.abs(distance_matrix(A, B) - C).max()
        print(error)

        blas_call2 = lambda: euclidean_dist_matrix(A, B, C)
        np_call = lambda : distance_matrix(A, B)

        print()
        # benchmark("blas1", blas_call)
        benchmark("blas2", blas_call2)
        benchmark("np", np_call)
        print()

    def test_mean_square_cols(self):

        C = np.random.randn(1005, 1000).astype(np.float32)

        row_values = np.zeros([C.shape[0]], dtype=np.float32)
        mean_square_cols(C, row_values)
        error = row_values - np.mean(C**2, 1)
        print("error:", np.max(error))

        blas = lambda: mean_square_cols(C, row_values)
        np_call = lambda : np.mean(C**2, 1)

        print()
        # benchmark("blas1", blas_call)
        benchmark("blas", blas)
        benchmark("np", np_call)
        print()

    def test_blas_match(self):

        A = np.random.randn(1000, 128).astype(np.float32)
        B = np.random.randn(900, 128).astype(np.float32)
        C = np.random.randn(1000, 900).astype(np.float32)

        euclidean_dist_matrix(A, B, C)
        row_indices = np.zeros([A.shape[0]], dtype=np.int32)
        col_indices = np.zeros([B.shape[0]], dtype=np.int32)
        argmin_match(C, row_indices, col_indices)

        dist = distance_matrix(A, B)

        print(np.abs( np.argmin(dist, 1) - row_indices).max())
        print(np.abs( np.argmin(dist, 0) - col_indices).max())

    def test_bf_cross_check_matcher(self):

        A = np.random.randint(0, 255, size=(1000, 128)).astype(np.float32)
        B = np.random.randint(0, 255, size=(1005, 128)).astype(np.float32)

        row_indices, col_indices = bf_cross_check_matcher(A, B)

        np_row_indices, np_col_indices = numpy_match(A, B)

        print(np.abs(np_row_indices - row_indices).max())
        print(np.abs(np_col_indices - col_indices).max())

        blas_call2 = lambda: bf_cross_check_matcher(A, B)
        np_call = lambda : numpy_match(A, B)

        print()
        # benchmark("blas1", blas_call)
        benchmark("blas2", blas_call2)
        benchmark("np", np_call)
        print()


def distance_matrix(X, Y):
    sqnorm1 = np.sum(np.square(X), 1, keepdims=True)
    sqnorm2 = np.sum(np.square(Y), 1, keepdims=True)
    innerprod = np.dot(X, Y.T)
    return sqnorm1 + np.transpose(sqnorm2) - 2.0 * innerprod


def numpy_match(X, Y):
    D = distance_matrix(X, Y)
    row_indices = np.argmin(D, 1)
    col_indices = np.argmin(D, 0)
    return row_indices, col_indices


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