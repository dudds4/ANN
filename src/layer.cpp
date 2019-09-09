#include "layer.h"
#include "neuralnet.h"
#include "activationTypes.h"
#include <exception>
#include "math_help.h"

double sigmoid(double x) 
{
	return 1.0f / (1 + exp(-1.0f * x));
}

void SigmoidLayer::activation(double *z, double *a)
{
	//TODO: optimize this loop
	for(int i = 0; i < n_outputs; ++i)
	{
		a[i] = sigmoid(z[i]);
	}
}

void SigmoidLayer::activationDeriv(const double *z, double *dz)
{
	if(!z || !dz) return;

	for(int i = 0; i < n_outputs; ++i)
	{
		double a = sigmoid(z[i]);
		dz[i] = a*(1-a);
	}
}


Layer::Layer(int i, int o) : n_inputs(i), n_outputs(o) {}
void Layer::forward_prop(const double *x, const double *params, double *z, double *a)
{
	if(!x || !params || !z || !a)
		return;

	const double *W = params;
	const double *b = params + (n_inputs*n_outputs);

	matrix_vector_mult(W, x, z, n_outputs, n_inputs);

	// add bias
	for(int r = 0; r < n_outputs; ++r)
		z[r] += b[r];

	activation(z, a);
}

void Layer::forward_prop(const NeuralNetMemory* mem)
{
	if(!mem)
		return;

	forward_prop(mem->inputs, mem->params, mem->activation_cache, mem->outputs);
}

void Layer::activationDeriv(const double *z, double *dz)
{
	if(!dz) return;

	for(int i = 0; i < n_outputs; ++i)
		dz[i] = 1;
}

void Layer::back_prop(const double* da, NeuralNetMemory* mem, double *dparams, double *dx)
{
	// compute dJ / dz
	double dz[n_outputs];
	activationDeriv(mem->activation_cache, dz);
	for(int i = 0; i < n_outputs; ++i)
		dz[i] *= da[i];

	double* dW = dparams;
	double* db = dparams + (n_outputs*n_inputs);

	// compute dW
	for(int i = 0; i < n_outputs; ++i)
	{
		for(int k = 0; k < n_inputs; ++k)
		{
			int index = i*n_inputs+k;
			dW[index] = mem->inputs[k] * dz[i];
		}
	}

	// compute db
	for(int i = 0; i < n_outputs; ++i)
		db[i] = dz[i];

	// compute dx
	matrix_transpose_vector_mult(mem->params, dz, dx, n_outputs, n_inputs);
}

int Layer::numParams() const
{
	return n_outputs*n_inputs + n_outputs;
}

void Layer::activation(double *z, double *a)
{
	//TODO: optimize this loop
	for(int i = 0; i < n_outputs; ++i)
	{
		a[i] = z[i];
	}
}