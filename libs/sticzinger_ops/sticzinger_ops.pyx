import numpy as np
cimport numpy as np
import cv2
from libcpp cimport bool

cimport cython

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint8

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint8_t DTYPE_t
ctypedef np.float32_t Float32_t
ctypedef np.float64_t Float64_t
ctypedef np.int32_t Int32_t

def fib(int n):
    """Print the Fibonacci series up to n."""
    cdef int a = 0
    cdef int b = 1

    while b < n:
        print(b)
        a, b = b, a + b

    print()



def uint8_array2d_to_ascii_v1(np.ndarray[DTYPE_t, ndim=2] array):
    return "".join([chr(x) for x in array.ravel()])


@cython.boundscheck(False)
@cython.wraparound(False)
def uint8_array2d_to_ascii_v2(DTYPE_t[:, :] array):
    cdef int i, j
    cdef list val = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val.append(chr(array[i, j]))

    return ''.join(val)


def uint8_array2d_to_ascii(np.ndarray[DTYPE_t, ndim=2] array):
    return array.tobytes().decode("ascii")



cpdef postprocess_and_refine_predictions(
        np.ndarray[Float32_t, ndim=2] predictions,
        list kp1,
        list kp2,
        float threshold = 10.0,
        float confidence = 0.99,
        bint homography_refine = True
):

    cdef np.ndarray[Int32_t, ndim=2] data = predictions.astype(np.int32)
    cdef np.ndarray[Int32_t, ndim=2] matches = data[data[:, 2] == 1][:, :2]

    cdef list cv_matches = [
        cv2.DMatch(_imgIdx=0, _queryIdx=q, _trainIdx=t, _distance=0)
        for q, t in matches
    ]

    cdef np.ndarray[Float64_t, ndim=2] H = None

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