#include "math_help.h"
#include <iostream>
using namespace std;

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

void matrix_transpose_vector_mult(const double *A, const double *x, double *y, int m, int n)
{
	//TODO: optimize this matrix multiplication
	for(int col = 0; col < n; ++col)
	{
		y[col] = 0;
		for(int row = 0; row < m; ++row)
		{
			int index = row*n + col;
			y[col] += A[index] * x[row];
		}
	}
}

