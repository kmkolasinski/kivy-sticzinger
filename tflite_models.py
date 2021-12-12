"""
https://github.com/teticio/kivy-tensorflow-helloworld/blob/main/buildozer.spec
"""
import numpy as np
from kivy.utils import platform

from logging_ops import measuretime

if platform == "android":
    from jnius import autoclass

    File = autoclass("java.io.File")
    Interpreter = autoclass("org.tensorflow.lite.Interpreter")
    InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
    Tensor = autoclass("org.tensorflow.lite.Tensor")
    DataType = autoclass("org.tensorflow.lite.DataType")
    TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
    TensorBufferFloat = autoclass(
        "org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat"
    )

    ByteBuffer = autoclass("java.nio.ByteBuffer")
    FloatBuffer = autoclass("java.nio.FloatBuffer")
    # Interpreter = autoclass("org.tensorflow.lite.Interpreter")
    CompatibilityList = autoclass("org.tensorflow.lite.gpu.CompatibilityList")
    GpuDelegate = autoclass("org.tensorflow.lite.gpu.GpuDelegate")

    from jnius import autoclass

    TestClass = autoclass("org.test.TestClass")


    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            model = File(model_filename)

            options = InterpreterOptions()

            # JVM exception occurred: Internal error: Failed to apply delegate: Attempting to use a delegate that only
            # supports static-sized tensors with a graph that has dynamic-sized tensors (tensor#21 is a
            # dynamic-sized tensor). java.lang.IllegalArgumentException

            # # https://www.tensorflow.org/lite/performance/gpu
            # compatList = CompatibilityList()
            # print("compatList.isDelegateSupportedOnThisDevice:", compatList.isDelegateSupportedOnThisDevice())
            # print("compatList.delegateOptions:", compatList.bestOptionsForThisDevice)
            #
            # delegateOptions = compatList.bestOptionsForThisDevice
            # options.addDelegate(GpuDelegate(delegateOptions))

            if num_threads is not None:
                options.setNumThreads(num_threads)

            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()
            self.not_allocated = True

        def allocate_tensors(self):
            print("Realocatting tensors")
            self.interpreter.allocateTensors()

            self.input_shape = self.interpreter.getInputTensor(0).shape()
            self.input_type = self.interpreter.getInputTensor(0).dataType()
            self.output_type = self.interpreter.getOutputTensor(0).dataType()

            self.output_shape = [self.input_shape[0], 3]
            self.output_buffer = TensorBuffer.createFixedSize(
                self.output_shape, self.output_type
            )
            self.input_buffer = TensorBufferFloat.createFixedSize(
                self.input_shape, self.input_type
            )

        def get_input_shape(self):
            return self.input_shape

        def resize_input(self, shape):
            if self.input_shape != list(shape):
                print(f"Resizing input shape to: {shape}")
                self.interpreter.resizeInput(0, shape)
                self.allocate_tensors()

        def pred(self, x):

            self.resize_input(x.shape)

            # xbytes = x.tobytes()
            # input = ByteBuffer.wrap(x.tobytes())

            if self.not_allocated:
                v = x.ravel().tolist()
                self.input_buffer.loadArray(v)
                self.not_allocated = False

            input = self.input_buffer.getBuffer()

            self.interpreter.run(input, self.output_buffer.getBuffer().rewind())

            # return np.reshape(np.array(self.output_buffer.getFloatArray()), self.output_shape)


else:
    import tensorflow as tf


    class TensorFlowModel:
        def load(self, model_filename, num_threads=None):
            self.interpreter = tf.lite.Interpreter(
                model_filename, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

        def resize_input(self, shape):
            if list(self.get_input_shape()) != shape:
                self.interpreter.resize_tensor_input(0, shape)
                self.interpreter.allocate_tensors()

        def get_input_shape(self):
            return self.interpreter.get_input_details()[0]["shape"]

        def pred(self, x):
            # assumes one input and one output for now
            self.interpreter.set_tensor(
                self.interpreter.get_input_details()[0]["index"], x
            )
            self.interpreter.invoke()
            return self.interpreter.get_tensor(
                self.interpreter.get_output_details()[0]["index"]
            )


class NpBFMatcher:
    def distance_matrix(self, X, Y):
        sqnorm1 = np.sum(np.square(X), 1, keepdims=True)
        sqnorm2 = np.sum(np.square(Y), 1, keepdims=True)
        innerprod = np.matmul(X, Y.T)
        return sqnorm1 + np.transpose(sqnorm2) - 2.0 * innerprod

    def match(self, dmat):
        row_matches = np.argmin(dmat, 1)
        col_matches = np.argmin(dmat, 0)

        num_rows = row_matches.shape[0]

        inverse_row_indices = col_matches[row_matches]
        row_indices = np.arange(0, num_rows, dtype=row_matches.dtype)

        cross_checked = row_indices == inverse_row_indices
        rows = row_indices[cross_checked]
        cols = row_matches[cross_checked]

        indices = np.transpose(np.stack([rows, cols]))

        distances = dmat[rows, cols]
        return indices, distances

    def __call__(self, X, Y):
        # X = X.astype(np.float16)
        # Y = Y.astype(np.float16)
        distm = self.distance_matrix(X, Y)
        distm = np.sqrt(np.maximum(distm, 0))
        indices, distances = self.match(distm)

        return indices, distances


def generate_random_XY(nx=512, ny=512, dim=256):
    X = np.random.randint(0, 255, size=(nx, dim))
    Y = np.random.randint(0, 255, size=(ny, dim))
    X = (X / 255).astype(np.float32)
    Y = (Y / 255).astype(np.float32)
    return X, Y


def benchmark(method, name, steps=10, warmup=1, **kwargs):
    with measuretime("warmup", log=False):
        for _ in range(warmup):
            X, Y = generate_random_XY(**kwargs)
            method(X, Y)

    with measuretime(name):
        for _ in range(steps):
            X, Y = generate_random_XY(**kwargs)
            method(X, Y)


def get_tf_matcher():
    import os

    model = TensorFlowModel()
    model.load(os.path.join(os.getcwd(), "model.tflite"))

    def tf_match(X, Y):
        values = model.pred(X)
        return values

    return tf_match


def get_np_matcher():
    model = NpBFMatcher()

    def tf_match(X, Y):
        values = model(X, X)
        return values

    return tf_match


def get_cv_matcher():
    import cv2

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def cv_match(X, Y):
        matches = bf.knnMatch(X, Y, k=1)
        matches_pairs = []
        distances = []
        for mm in matches:
            if len(mm) == 1:
                match = mm[0]
                row, col = match.queryIdx, match.trainIdx
                matches_pairs.append([row, col])
                distances.append(match.distance)
        return matches_pairs, distances

    return cv_match


import os
from sticzinger_ops import uint8_array2d_to_ascii, postprocess_and_refine_predictions
import numpy as np
import cv2
from matching import match_images


class TFLiteBFMatcher:
    def __init__(self, path, num_threads=1, xnn=False, homography_refine=True):
        tflite_model = TestClass()
        path = os.path.join(os.getcwd(), path)
        tflite_model.loadModel(path, num_threads, xnn)

        self.model = tflite_model
        self.homography_refine = homography_refine

        self.X_str_cache = None
        self.X_id = None

    def match(self, kp1, des1, kp2, des2, ransack_threshold: float = 10):

        X_id = id(des1)
        if X_id != self.X_id:
            # sift returns values from 0 - 255 we support only values from 0 - 127
            X = des1.astype(np.uint8) // 2
            X_str = uint8_array2d_to_ascii(X)
            self.X_id = X_id
            self.X_str_cache = X_str
        else:
            X = des1
            X_str = self.X_str_cache

        Y = des2.astype(np.uint8) // 2
        Y_str = uint8_array2d_to_ascii(Y)

        self.model.resizeInputs(X.shape, Y.shape)
        predictions = self.model.predict(X_str, Y_str)
        predictions = np.reshape(np.array(predictions), self.model.getOutputShape())
        predictions = predictions.astype(np.float32)
        with measuretime("cython-postprocessing"):
            H, matches = postprocess_and_refine_predictions(
                predictions, kp1, kp2, homography_refine=self.homography_refine
            )

        with measuretime("py-postprocessing"):
            mask = predictions[:, 2].astype(np.int32) == 1
            matches = predictions[mask][:, :2].astype(np.int32)

            matches = [
                cv2.DMatch(_imgIdx=0, _queryIdx=q, _trainIdx=t, _distance=0)
                for q, t in matches
            ]

            H = None
            if self.homography_refine:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                )
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                )
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, ransack_threshold, confidence=0.99
                )
                matches_mask = mask.ravel()
                matches = [
                    match
                    for match, mask_val in zip(matches, matches_mask)
                    if mask_val == 1
                ]

        return H, matches


class CvBFMatcher:
    def __init__(self, homography_refine=True):
        self.homography_refine = homography_refine

    def match(self, kp1, des1, kp2, des2):
        return match_images(
            kp1,
            des1,
            kp2,
            des2,
            bf_matcher_cross_check=True,
            bf_matcher_norm="NORM_L2",
            ransack_threshold=10.0,
            matcher_type="brute_force",
            homography_refine=self.homography_refine,
        )


def benchmark_tflite_model(name, matcher, steps: int = 5):
    with measuretime(name):
        X = np.random.randint(0, 256, [1024, 128]).astype(np.float32)
        for i in range(steps):
            Y = np.random.randint(0, 256, [2000 + i, 128]).astype(np.float32)
            _, matches = matcher.match([], X, [], Y)


def benchmark_scenarios():
    # import requests
    # with open('model.tflite', "wb") as f:
    #     data = requests.get("http://0.0.0.0:3333/model.tflite").content
    #     f.write(data)

    matcher = TFLiteBFMatcher("model.tflite", homography_refine=False)
    benchmark_tflite_model("tf-matcher", matcher)

    matcher = TFLiteBFMatcher("model-dq.tflite", homography_refine=False)
    benchmark_tflite_model("tf-matcher-dq", matcher)

    matcher = TFLiteBFMatcher("model-dq.tflite", homography_refine=False, num_threads=2)
    benchmark_tflite_model("tf-matcher-dq 2 threard", matcher)

    matcher = TFLiteBFMatcher("model-fq.tflite", homography_refine=False)
    benchmark_tflite_model("tf-matcher-fq", matcher)

    matcher = CvBFMatcher(homography_refine=False)
    benchmark_tflite_model("cv-matcher", matcher)
