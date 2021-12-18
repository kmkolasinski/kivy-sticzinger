#include "cblas.h"
#include <immintrin.h>

// https://github.com/michael-lehn/ulmBLAS/tree/master/interfaces/blas/C


int argmin_vector(int *n, float *sx){

    int ret_val;
    float smin, r__1;
    int i;

    ret_val = 0;
    smin = sx[0];

    for (i = 0; i < *n; ++i) {
        r__1 = sx[i];
        if (r__1 < smin) {
            smin = r__1;
            ret_val = i;
        }
    }
    return ret_val;

}

float vector_sq_mean(float *x, int n){

    float ret_val;
    int i;

    ret_val = 0;

    for (i = 0; i < n; ++i) {
        ret_val += x[i] * x[i];
    }
    return ret_val / n ;

}


int isamin(int *n, float *sx, int *incx){

    //# https://github.com/xiaoyeli/superlu/blob/master/CBLAS/idamax.c
    /* System generated locals */
    int ret_val;
    float r__1;

    /* Local variables */
    float smin;
    int i, ix;


    /* finds the index of element having max. absolute value.
       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*)

    */

    ret_val = 0;

    if (*n < 1 || *incx <= 0) {
	    return ret_val;
    }

    if (*n == 1) {
	    return ret_val;
    }

    if (*incx == 1) {
	    goto L20;
    }

    /*        code for increment not equal to 1 */
    ix = 0;
    smin = sx[0];
    ix += *incx;
    for (i = 1; i < *n; ++i) {
        if ((r__1 = sx[ix], r__1 > smin)) {
            goto L5;
        }
        ret_val = i;
        smin = (r__1 = sx[ix], r__1);
L5:
	    ix += *incx;
    }
    return ret_val;

    /*        code for increment equal to 1 */

L20:

    smin = sx[0];
    for (i = 1; i < *n; ++i) {
        if ((r__1 = sx[i], r__1 > smin)) {
            goto L30;
        }
        ret_val = i;
        smin = (r__1 = sx[i], r__1);
L30:
	;
    }
    return ret_val;
}


void sum_square_cols(float* X, float *y, int num_rows, int num_cols) {

  int i, offset;

  for (i = 0; i < num_rows; ++i)
  {
       offset = i * num_cols;
       y[i] = cblas_sdot(num_cols, (X + offset), 1, (X + offset), 1);
  }

}

void argmin_col(float* X, int *y, int num_rows, int num_cols) {
  int i, inc;
  inc = 1;
  for (i = 0; i < num_rows; ++i)
  {
       y[i] = argmin_vector(&num_cols, (X + i * num_cols));
       // y[i] = isamin(&num_cols, (X + i * num_cols), &inc);
  }
}


void argmin_row(float* X, int *y, int num_rows, int num_cols) {
  int j;
  for (j = 0; j < num_cols; ++j)
  {
       y[j] = isamin(&num_rows, (X + j), &num_cols);
  }
}



inline void sum_row_and_col_vectors(float* row, float *col, float* X, int num_rows, int num_cols) {

  int i, j;
  float *row_ptr;
  float row_val;
  for (i = 0; i < num_rows; ++i){

    row_ptr = (X + i * num_cols);
    row_val = row[i];

    for (j = 0; j < num_cols; ++j){
       row_ptr[j] = row[i] + col[j];
    }
  }
}