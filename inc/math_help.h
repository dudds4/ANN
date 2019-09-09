
#ifndef MATH_HELP
#define MATH_HELP

void matrix_vector_mult(const double *A, const double *x, double *y, int m, int n);
void matrix_transpose_vector_mult(const double *A, const double *x, double *y, int m, int n);
void printMatrix(const double* A, int m, int n);

#endif