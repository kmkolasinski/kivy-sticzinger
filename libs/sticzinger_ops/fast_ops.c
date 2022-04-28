#include "cblas.h"

// https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
//
// https://github.com/DLTcollab/sse2neon

#ifdef LINUX
    #include <immintrin.h>
#else
    #include "sse2neon.h"
#endif

#include "fast_ops.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//#define PMIN(x, y) (((*x) < (*y)) ? (x) : (y))

//// https://github.com/michael-lehn/ulmBLAS/tree/master/interfaces/blas/C
//int _mm256_hmax_index(const __m256i v)
//{
//    __m256i vmax = v;
//
//    vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
//    vmax = _mm256_max_epu32(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
//    vmax = _mm256_max_epu32(vmax, _mm256_permute2x128_si256(vmax, vmax, 0x01));
//
//    __m256i vcmp = _mm256_cmpeq_epi32(v, vmax);
//
//    int mask = _mm256_movemask_epi8(vcmp);
//
//    return __builtin_ctz(mask) >> 2;
//}

//int find_max(int* buff, int size)
//{
// //https://3way.com.ar/2019/01/17/using-gcc-intrinsics-mmx-ssex-avx-to-look-for-max-value-in-array/
//    int maxmax[8];
//    int i;
//    int max = buff[0];
//
//    __m128i *f8 = (__m128i*)buff;
//    __m128i maxval = _mm_setzero_si128();
//    __m128i maxval2 = _mm_setzero_si128();
//    for (i = 0; i < size / 16; i++) {
//        maxval = _mm_max_epi16(maxval, f8[i]);
//    }
//    maxval2 = maxval;
//    for (i = 0; i < 3; i++) {
//        maxval = _mm_max_epi16(maxval, _mm_shufflehi_epi16(maxval, 0x3));
//        _mm_store_si128(&maxmax, maxval);
//        maxval2 = _mm_max_epi16(maxval2, _mm_shufflelo_epi16(maxval2, 0x3));
//        _mm_store_si128(&maxmax, maxval2);
//    }
//    _mm_store_si128(&maxmax, maxval);
//    for(i = 0; i < 8; i++)
//        if(max < maxmax[i])
//            max = maxmax[i];
//    return max;
//}

void printv(__m128 v){
    printf("fvec = [ %f, %f, %f, %f ]\n", v[0], v[1], v[2], v[3]);
}

void printvi(__m128i v){
    int v0 = _mm_extract_epi32(v, 0);
    int v1 = _mm_extract_epi32(v, 1);
    int v2 = _mm_extract_epi32(v, 2);
    int v3 = _mm_extract_epi32(v, 3);

    printf("ivec = [ %d, %d, %d, %d ]\n", v0, v1, v2, v3);
}


float find_min(float* buff, int n)
{
    // https://doc.rust-lang.org/nightly/core/arch/x86_64/index.html
    // https://www.cs.virginia.edu/~cr4bd/3330/S2018/simdref.html
    int i;
    float vmin = buff[0];
    const int K = 4;
    __m128 minval = _mm_loadu_ps(&buff[0]);

    for (i = 0; i + K < n; i += K) {
         minval = _mm_min_ps(minval,  _mm_loadu_ps(&buff[i]));
    }

//    printf("end i = %d \n", i);
    for (; i < n; ++i) {
//        printf("rest i = %d \n", i);
        if(buff[i] < vmin){
            vmin = buff[i];
        }
    }

    for(i = 0; i < K; ++i){
        if(minval[i] < vmin){
            vmin = minval[i];
        }
    }

    return vmin;
}

float find_smallest_element_in_matrix_SSE(const float* m, int n, int *minIndex)
{
    // https://stackoverflow.com/questions/33844471/find-largest-element-in-matrix-and-its-column-and-row-indexes-using-sse-and-avx
    const int K = 4;
    float minVal = m[0];
    float aMinVal[K];
    int aMinIndex[K];
    int i, k;


    const __m128i vIndexInc = _mm_set1_epi32(K);
    __m128i vMinIndex = _mm_setr_epi32(0, 1, 2, 3);
    __m128i vIndex = vMinIndex;
    __m128 vMinVal = _mm_loadu_ps(m);

//    printvi(vIndex);

    for (i = 0; i + K < n; i += K)
    {
        __m128 v = _mm_loadu_ps(&m[i]);
        __m128 vcmp = _mm_cmple_ps(v, vMinVal);

        vMinVal = _mm_blendv_ps(vMinVal, v, vcmp);

        vMinIndex = _mm_blendv_epi8(vMinIndex, vIndex, _mm_castps_si128(vcmp));
        vIndex = _mm_add_epi32(vIndex, vIndexInc);
    }

//    printv(vMinVal);
//    printvi(vMinIndex);

    _mm_storeu_ps(aMinVal, vMinVal);
    _mm_storeu_si128((__m128i *)aMinIndex, vMinIndex);

    minVal = aMinVal[0];
    *minIndex = aMinIndex[0];

    for (k = 1; k < K; ++k)
    {
        if (aMinVal[k] < minVal)
        {
            minVal = aMinVal[k];
            *minIndex = aMinIndex[k];
        }
    }

    for (; i < n; ++i) {
        if(m[i] < minVal){
            minVal = m[i];
            *minIndex = i;
        }
    }

    return minVal;
}



int argmin_vector(float *x, int n, float* min_value){

    int ret_val;
    float smin, val;
    float *pval;
    int i, k, p;

    ret_val = 0;
    smin = x[0];
    pval = x;

    smin = find_min(x, n);

//    for (i = 0; i < n; ++i) {
//        if(x[i] == smin){
//            break;
//        }
//    }
//    ret_val = i ;
//    printf("min index: %d \n", i);

    const int K = 4;
    const __m128i vIndexInc = _mm_set1_epi32(K);
    const __m128 vMinVal = _mm_set1_ps(smin);

    __m128i vMinIndex = _mm_setr_epi32(0, 1, 2, 3);
    __m128i vIndex = vMinIndex;

    for (i = 0; i + 4 < n; i+=4) {
        __m128 vcmp = _mm_cmpeq_ps(_mm_loadu_ps(&x[i]), vMinVal);
        __m128i mask = _mm_castps_si128(vcmp);
        vMinIndex = _mm_min_epi32(_mm_madd_epi16(vIndex, mask), vMinIndex);
        vIndex = _mm_add_epi32(vIndex, vIndexInc);
    }
//    printvi(vMinIndex);

    k = -1;
    for (; i < n; ++i) {
        k = (x[i] == smin) ? i : k;
    }

    if ( k < 0){
        k = MAX(-_mm_extract_epi32(vMinIndex, 0), k);
        k = MAX(-_mm_extract_epi32(vMinIndex, 1), k);
        k = MAX(-_mm_extract_epi32(vMinIndex, 2), k);
        k = MAX(-_mm_extract_epi32(vMinIndex, 3), k);
    }
//    printf("min index 2b: %d \n", k);
    ret_val = k;
    *min_value = smin;
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

  int i, j, offset;
  float sum;
  float *row_ptr;

  for (i = 0; i < num_rows; ++i)
  {
       offset = i * num_cols;
       row_ptr = (X + i * num_cols);
       sum = 0.0;
       for (j = 0; j < num_cols; ++j){
            sum += row_ptr[j] * row_ptr[j];
       }
       y[i] = sum;
  }

}

void argmin_col(float* X, int *y, int num_rows, int num_cols) {
  int i, inc;
  float min_value;
  inc = 1;
  for (i = 0; i < num_rows; ++i)
  {
       y[i] = argmin_vector((X + i * num_cols), num_cols, &min_value);
//        y[i] = isamin(&num_cols, (X + i * num_cols), &inc);
  }
}


void argmin_row(float* X, int *y, int num_rows, int num_cols) {
  int j;
  for (j = 0; j < num_cols; ++j)
  {
       y[j] = isamin(&num_rows, (X + j), &num_cols);
  }
}


void fast_cross_check_match(int *irow, float *vrow, float *vcol, float* X, int num_rows, int num_cols) {

  int i, j;
  float min_value;
  float *row_ptr;

  for (i = 0; i < num_rows; ++i){
       irow[i] = argmin_vector((X + i * num_cols), num_cols, &min_value);
       vrow[i] = min_value;
  }

  #pragma GCC ivdep
  for (j = 0; j < num_cols; ++j){
      vcol[j] = X[j];
  }

  for (i = 0; i < num_rows; ++i){
    row_ptr = (X + i * num_cols);

    #pragma GCC ivdep
    for (j = 0; j < num_cols; ++j){
        vcol[j] = MIN(row_ptr[j], vcol[j]);
    }
  }

}


void sum_row_and_col_vectors(float* row, float *col, float* X, int num_rows, int num_cols) {

  int i, j;
  float *row_ptr;
  float row_val;
  for (i = 0; i < num_rows; ++i){

    row_ptr = (X + i * num_cols);
    row_val = row[i];

    #pragma GCC ivdep
    for (j = 0; j < num_cols; ++j){
       row_ptr[j] = row_val + col[j];
    }
  }
}


void sum_row_and_col_vectors_v2(float* row, float *col, float* X, int num_rows, int num_cols) {

  int i, j;
  float *row_ptr;
  float row_val;
  for (i = 0; i < num_rows; ++i){

    row_ptr = (X + i * num_cols);
    row_val = row[i];

    #pragma GCC ivdep
    for (j = 0; j < num_cols; ++j){
       row_ptr[j] = -2 *row_ptr[j] + row_val + col[j];
    }
  }
}