
int argmin_vector(float *x, int n, float* v);
float vector_sq_mean(float *x, int n);
int isamin(int *n, float *sx, int *incx);
void sum_square_cols(float* X, float *y, int num_rows, int num_cols) ;
void argmin_col(float* X, int *y, int num_rows, int num_cols) ;
void argmin_row(float* X, int *y, int num_rows, int num_cols) ;
void sum_row_and_col_vectors(float* row, float *col, float* X, int num_rows, int num_cols) ;
void sum_row_and_col_vectors_v2(float* row, float *col, float* X, int num_rows, int num_cols);
void fast_cross_check_match(int *irow, float *vrow, float *vcol, float* X, int num_rows, int num_cols);