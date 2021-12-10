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
    TensorBufferFloat = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat")

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
            self.output_buffer = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
            self.input_buffer = TensorBufferFloat.createFixedSize(self.input_shape, self.input_type)

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


class NpBFMatcher():

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
    model.load(os.path.join(os.getcwd(), 'model.tflite'))

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
        matches = bf.knnMatch(X, X, k=1)
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