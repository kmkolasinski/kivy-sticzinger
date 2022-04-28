# distutils: sources = fast_ops.c

import numpy as np
cimport numpy as np
import cv2
cimport cython
from libc.stdlib cimport malloc, free

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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef euclidean_distance_matrix(float[:, :] X, float[:, :] Y):
    cdef float[:, :] sqnorm1 = np.sum(np.square(X), 1, keepdims=True)
    cdef float[:, :] sqnorm2 = np.sum(np.square(Y), 1, keepdims=True)
    cdef float[:, :] innerprod = 2 * np.dot(X, Y.T)
    return sqnorm1 + np.transpose(sqnorm2) - innerprod


ctypedef int CBLAS_INDEX


cdef extern from 'cblas.h':

    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans

    ctypedef enum CBLAS_LAYOUT:
        CblasRowMajor
        CblasColMajor

    ctypedef enum CBLAS_UPLO:
        CblasUpper
        CblasLower


    void lib_sgemm "cblas_sgemm"(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc) nogil

    void simple_sgemm "sgemm"(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, int M, int N, int K,
                             float  alpha, float  *A, int lda, float  *B, int ldb,
                             float  beta, float  *C, int ldc) nogil


cdef extern from "fast_ops.h":
    void sum_square_cols(float* X, float *y, int num_rows, int num_cols) nogil
    void sum_row_and_col_vectors(float * row, float *col, float* X, int num_rows, int num_cols) nogil
    void sum_row_and_col_vectors_v2(float * row, float *col, float* X, int num_rows, int num_cols) nogil
    void argmin_row(float * X, int *y, int num_rows, int num_cols) nogil
    void argmin_col(float * X, int *y, int num_rows, int num_cols) nogil
    float vector_sq_mean(float *X, int n) nogil
    void fast_cross_check_match(int *irow, float *vrow, float *vcol, float * X, int num_rows, int num_cols) nogil


cdef extern  from "ulmblas.h":
    void dgemm_nn(int            m,
                  int            n,
                  int            k,
                  double         alpha,
                  const double *A,
                  int            incRowA,
                  int            incColA,
                  const double *B,
                  int            incRowB,
                  int            incColB,
                  double         beta,
                  double *C,
                  int            incRowC,
                  int            incColC)


cpdef void ulm_dgemm(double alpha, double[:, ::1] A, double[:, ::1] B, double beta, double[:, ::1] C):

    """
     C = α A B^T + β C
    """

    cdef double* A_ptr = &A[0, 0]
    cdef double* B_ptr = &B[0, 0]
    cdef double* C_ptr = &C[0, 0]

    dgemm_nn(C.shape[0], C.shape[1], A.shape[1], alpha, A_ptr, A.shape[1], 1, B_ptr,
               B.shape[1], 1, beta, C_ptr, C.shape[1], 1)


cpdef void sgemm5v3(float alpha, float[:, ::1] A, float[:, ::1] B,
                      float beta, float[:, ::1] C):

    """
     C = α A B^T + β C
    """


    cdef float* A_ptr = &A[0, 0]
    cdef float* B_ptr = &B[0, 0]
    cdef float* C_ptr = &C[0, 0]

    lib_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
               C.shape[0], C.shape[1],
               A.shape[1], alpha, A_ptr, A.shape[1], B_ptr,
               B.shape[1], beta, C_ptr, C.shape[1])

    # simple_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
    #            C.shape[0], C.shape[1],
    #            A.shape[1], alpha, A_ptr, A.shape[1], B_ptr,
    #            B.shape[1], beta, C_ptr, C.shape[1])



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void euclidean_dist_matrix(float[:, ::1] A, float[:, ::1] B, float[:, ::1] C):


    cdef float* A_ptr = &A[0, 0]
    cdef float* B_ptr = &B[0, 0]
    cdef float* C_ptr = &C[0, 0]

    cdef int a_num_rows = A.shape[0]
    cdef int a_num_cols = A.shape[1]
    cdef int b_num_rows = B.shape[0]
    cdef int b_num_cols = B.shape[1]

    cdef float *a_sq = <float*> malloc(a_num_rows * sizeof(float))
    cdef float *b_sq = <float*> malloc(b_num_rows * sizeof(float))

    try:
        sum_square_cols(A_ptr, a_sq, a_num_rows, a_num_cols)
        sum_square_cols(B_ptr, b_sq, b_num_rows, b_num_cols)
        sum_row_and_col_vectors(a_sq, b_sq, C_ptr, a_num_rows, b_num_rows)
        sgemm5v3(-2, A, B, 1, C)

    finally:
        free(a_sq)
        free(b_sq)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void argmin_match(float[:, ::1] X, int[::1] row_indices, int[::1] col_indices):

    cdef float* X_ptr = &X[0, 0]
    cdef:
        int num_rows = row_indices.shape[0]
        int num_cols = col_indices.shape[0]

    argmin_col(X_ptr, &row_indices[0], num_rows, num_cols)
    argmin_row(X_ptr, &col_indices[0], num_rows, num_cols)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void column_argmin(float[:, ::1] X, int[::1] row_indices):

    cdef:
        float * X_ptr = &X[0, 0]
        int num_rows = X.shape[0]
        int num_cols = X.shape[1]

    argmin_col(X_ptr, &row_indices[0], num_rows, num_cols)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mean_square_cols(float[:, ::1] A, float[::1] y):


    cdef float* A_ptr = &A[0, 0]

    cdef int a_num_rows = A.shape[0]
    cdef int a_num_cols = A.shape[1]

    for i in range(a_num_rows):
        y[i] = vector_sq_mean(&A[i, 0], a_num_cols)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bf_cross_check_matcher(float[:, ::1] A, float[:, ::1] B):

    cdef:
        int num_rows = A.shape[0]
        int num_cols = B.shape[0]

    cdef float[:,::1] C = np.zeros((num_rows, num_cols), dtype = np.float32)
    cdef int[::1] row_indices = np.zeros((num_rows,), dtype = np.int32)

    euclidean_dist_matrix(A, B, C)

    cdef float* C_ptr = &C[0, 0]

    cdef float[::1] row_values = np.zeros((num_rows,), dtype = np.float32)
    cdef float[::1] col_values = np.zeros((num_cols,), dtype = np.float32)

    fast_cross_check_match(
        &row_indices[0], &row_values[0], &col_values[0],
        C_ptr, num_rows, num_cols
    )

    row_index = np.arange(0, A.shape[0])
    cross_checked = row_values == np.array(col_values)[row_indices]
    rows = row_index[cross_checked]
    cols = np.array(row_indices)[cross_checked]
    distances = np.array(row_values)[cross_checked]
    indices = np.transpose(np.stack([rows, cols]))

    return indices, distances


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void euclidean_dist_matrix_v2(float[:, ::1] A, float[:, ::1] B, float[:, ::1] C):


    cdef float* A_ptr = &A[0, 0]
    cdef float* B_ptr = &B[0, 0]
    cdef float* C_ptr = &C[0, 0]

    cdef int a_num_rows = A.shape[0]
    cdef int a_num_cols = A.shape[1]
    cdef int b_num_rows = B.shape[0]
    cdef int b_num_cols = B.shape[1]

    cdef float *a_sq = <float*> malloc(a_num_rows * sizeof(float))
    cdef float *b_sq = <float*> malloc(b_num_rows * sizeof(float))

    try:
        sum_square_cols(A_ptr, a_sq, a_num_rows, a_num_cols)
        sum_square_cols(B_ptr, b_sq, b_num_rows, b_num_cols)
        sum_row_and_col_vectors_v2(a_sq, b_sq, C_ptr, a_num_rows, b_num_rows)
        #sgemm5v3(-2, A, B, 1, C)

    finally:
        free(a_sq)
        free(b_sq)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bf_cross_check_matcher_v2(
        np.ndarray[Float32_t, ndim=2] A, np.ndarray[Float32_t, ndim=2] B):

    cdef:
        int num_rows = A.shape[0]
        int num_cols = B.shape[0]

    # cdef float[:,::1] C = np.zeros((num_rows, num_cols), dtype = np.float32)
    cdef int[::1] row_indices = np.zeros((num_rows,), dtype = np.int32)

    # this is slow on android !!!
    cdef np.ndarray[Float32_t, ndim=2] C = np.ascontiguousarray(np.matmul(A, np.transpose(B)))
    # cdef np.ndarray[Float32_t, ndim=2] C = np.zeros([num_rows, num_cols], np.float32)

    euclidean_dist_matrix_v2(A, B, C)

    cdef float* C_ptr = <float*> C.data #[0, 0]

    cdef float[::1] row_values = np.zeros((num_rows,), dtype = np.float32)
    cdef float[::1] col_values = np.zeros((num_cols,), dtype = np.float32)

    fast_cross_check_match(
        &row_indices[0], &row_values[0], &col_values[0],
        C_ptr, num_rows, num_cols
    )

    row_index = np.arange(0, A.shape[0])
    cross_checked = row_values == np.array(col_values)[row_indices]
    rows = row_index[cross_checked]
    cols = np.array(row_indices)[cross_checked]
    distances = np.array(row_values)[cross_checked]
    indices = np.transpose(np.stack([rows, cols]))

    return indices, distances