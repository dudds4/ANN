#include "layer.h"
#include "activationTypes.h"
#include <exception>


double sigmoid(double x) 
{
	return 1.0f / (1 + exp(-1.0f * x));
}

void SigmoidLayer::activation(double *z, double *a, double *d, int n)
{
	//TODO: optimize this loop
	for(int i = 0; i < n; ++i)
	{
		a[i] = sigmoid(z[i]);
		d[i] = a[i] * (1 - a[i]);
	}
}

Layer::Layer(int i, int o) : n_inputs(i), n_outputs(o) {}

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

void Layer::forward_prop(const double *x, const double *params, double *partials, double *a)
{
	if(!x || !params || !a)
		return;

	double z[n_outputs] = {0};

	const double *W = params;
	const double *b = params + (n_inputs*n_outputs);

	matrix_vector_mult(W, x, z, n_outputs, n_inputs);

	// add bias
	for(int r = 0; r < n_outputs; ++r)
		z[r] += b[r];

	double da_by_dz[n_outputs];
	activation(z, a, da_by_dz, n_outputs);

	if(partials)
	{
		double *W_partial = partials;
		double *b_partial = partials + (n_inputs*n_outputs);

		for(int i = 0; i < n_outputs; ++i)
		{
			for(int k = 0; k < n_inputs; ++k)
			{
				W_partial[i*n_outputs + k] = x[k] * da_by_dz[i];
			}
		}

		for(int i = 0; i < n_outputs; ++i)
			b_partial[i] = da_by_dz[i];

	}
}


int Layer::numParams() const
{
	return n_outputs*n_inputs + n_outputs;
}

void Layer::activation(double *z, double *a, double *d, int n)
{
	//TODO: optimize this loop
	for(int i = 0; i < n; ++i)
	{
		a[i] = z[i];
		d[i] = 1;
	}
}

// void Layer::setParent(Layer* p)
// {
// 	if(this == p) { throw 1; }
// 	if(p.n_outputs != this->n_inputs) { throw 2; }

// }

// Layer* generateLayer(int i, int o, int a)
// {

// }
