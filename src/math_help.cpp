// compute y = Ax, where A is an m x n matrix
void matrix_vector_mult(const double *A, const double *x, double *y, int m, int n)
{
	//TODO: optimize this matrix multiplication
	for(int r = 0; r < m; ++r)
	{
		y[r] = 0;
		for(int c = 0; c < n; ++c)
		{
			y[r] += A[r*n + c] * x[c];
		}
	}	
}
#include <iostream>
using namespace std;

void matrix_transpose_vector_mult(const double *A, const double *x, double *y, int m, int n)
{
	//TODO: optimize this matrix multiplication
	for(int j = 0; j < m; ++j)
	{
		y[j] = 0;

		for(int i = 0; i < n; ++i)
		{
			y[j] += A[j*n + i] * x[i];
		}
	}
}

#include <iostream>
void printMatrix(const double* A, int m, int n)
{

	for(int r = 0; r < m; ++r)
	{
		for(int c = 0; c < n; ++c)
		{
			std::cout <<  A[r*n + c] << " ";
		}
		std::cout << std::endl;
	}
}