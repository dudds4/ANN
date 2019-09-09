#include "layer.h"
#include "neuralnet.h"
#include "math_help.h"
#include <time.h>

#include <iostream>

using namespace std;

#define ABS(x) ((x) > 0 ? (x) : -(x))
#define ASSERT_VECTOR_EQUALS(vec, len, val) do {			\
	for(int i = 0; i < len; ++i)							\
		if(ABS(vec[i] - val) > 0.00001)						\
			return 1;										\
} while(0)

#define ASSERT_EQUAL(a, b) do {			\
	if(ABS(a - b) > 0.00001)			\
	{									\
		cout << "Assert equal failed: " << a << " != " << b << endl; \
		return 1;						\
	}									\
} while(0)

double measure_param_derivative(Layer* l, NeuralNetMemory* mem, int param_index, int r)
{
	const double dp = 0.00000001;

	l->forward_prop(mem);

	double out1 = mem->outputs[r];

	double temp = mem->params[param_index];
	mem->params[param_index] += dp;

	l->forward_prop(mem);
	double out2 = mem->outputs[r];

	mem->params[param_index] = temp;

	return (out2 - out1) / dp;
}

double measure_input_derivative(Layer* l, NeuralNetMemory* mem, int input_index, int r)
{
	const double dp = 0.00000001;

	l->forward_prop(mem);

	double out1 = mem->outputs[r];

	double temp = mem->inputs[input_index];
	mem->inputs[input_index] += dp;

	l->forward_prop(mem);
	double out2 = mem->outputs[r];

	mem->inputs[input_index] = temp;

	return (out2 - out1) / dp;
}

int test_general_layer(Layer *l, NeuralNetMemory *mem)
{
	// should fail but not crash
	l->forward_prop(nullptr, mem->params, mem->activation_cache, mem->outputs);
	l->forward_prop(mem->inputs, nullptr, mem->activation_cache, mem->outputs);
	l->forward_prop(mem->inputs, mem->params, nullptr, mem->outputs);
	l->forward_prop(mem->inputs, mem->params, mem->activation_cache, nullptr);

	mem->randomizeParams();

	// run forward prop
	l->forward_prop(mem);

	// run backward prop
	int n_out = l->n_outputs;
	int n_in = l->n_inputs;
	int n_params = n_out * n_in + n_out;

	double da[n_out];
	for(int i = 0; i < n_out; ++i)
		da[i] = 1;

	double dparams[n_params];
	double dinputs[l->n_inputs];

	l->back_prop(da, mem, dparams, dinputs);
	// printMatrix(dparams, n_out, n_in);

	// check the partials for W
	for(int r = 0; r < n_out; ++r)
	{
		for(int c = 0; c < n_in; ++c)
		{
			int index = r*n_in + c;
			double deriv = measure_param_derivative(l, mem, index, r);
			ASSERT_EQUAL(dparams[index], deriv);
		}
	}

	// // check the partials for b
	for(int r = 0; r < n_out; ++r)
	{
		int index = n_out*n_in + r;
		double deriv = measure_param_derivative(l, mem, index, r);
		ASSERT_EQUAL(dparams[index], deriv);
	}

	// check the partials for x

	// or dont.?

	// for(int i = 0; i < n_in; ++i)
	// {
	// 	double sum = 0;
	// 	cout << endl;
	// 	for(int o = 0; o < n_out; ++o)
	// 	{
	// 		double deriv = measure_input_derivative(l, mem, i, o);
	// 		// cout << "d" << i << "/d" << j << "=" << deriv << "=" << mem->params[i*n_out + j] << ", ";
	// 		sum += deriv;
	// 	}


	// 	// cout << endl;
	// 	cout << sum << " " << dinputs[i] << " ";
	// }

	// cout << endl;

	// for(int i = 0; i < n_in; ++i)
	// 	cout << dinputs[i] << ", ";

	// cout << endl;

	return 0;
}

int test_layer()
{
	// a plain layer is just a 
	// matrix multiplication and vector addition

	int n_in = 3, n_out = 2;
	int n_params = n_in*n_out + n_out;

	NeuralNetMemory mem;
	mem.allocate(n_in, n_out, n_params);

	Layer l(n_in,n_out);
	
	// input x
	mem.inputs[0] = -5;
	mem.inputs[1] = -13;
	mem.inputs[2] = 7;

	// W identity matrix
	mem.params[0] = 2;
	mem.params[1] = -7;
	mem.params[2] = 3;
	mem.params[3] = 11;
	mem.params[4] = 3;
	mem.params[5] = 11;

	// b
	mem.params[6] = -101;
	mem.params[7] = 18;

	l.forward_prop(&mem);
	ASSERT_VECTOR_EQUALS(mem.outputs, n_out, 1);

	return test_general_layer(&l, &mem);
}

int sigmoid_layer()
{
	// a plain layer is just a 
	// matrix multiplication and vector addition

	int n_in = 2, n_out = 2;
	int n_params = n_in*n_out + n_out;

	NeuralNetMemory mem;
	mem.allocate(n_in, n_out, n_params);

	SigmoidLayer l(n_in,n_out);
	
	// input x
	mem.inputs[0] = -5;
	mem.inputs[1] = -13;

	// W identity matrix
	mem.params[0] = 2;
	mem.params[1] = -7;
	mem.params[2] = 3;
	mem.params[3] = 11;

	// b
	mem.params[4] = -81;
	mem.params[5] = 158;

	// should run without computing partial derivatives
	l.forward_prop(&mem);
	ASSERT_VECTOR_EQUALS(mem.outputs, n_out, 0.5);

	mem.params[4] = 100;
	mem.params[5] = -100;	

	// should run computing partial derivatives
	l.forward_prop(&mem);
	ASSERT_EQUAL(mem.outputs[0], 1);
	ASSERT_EQUAL(mem.outputs[1], 0);

	return test_general_layer(&l, &mem);

}


// matrix_vector_mult(const double *A, const double *x, double *y, int m, int n)
int matrix_mult()
{
	{
		int m = 2, n = 3;
		double A[m*n] = {1, 2, 3, 4, 5, 6};
		double x[n] = {-2, 3, -5};
		double expected[m] = {-11, -23};
		double actual[m];
		matrix_vector_mult(A, x, actual, m, n);

		for(int i = 0; i < m; ++i)
			ASSERT_EQUAL(actual[i], expected[i]);
	}

	{
		int m = 3, n = 2;
		double A[m*n] = {1,4,2,5,3,6};
		double x[n] = {-2, 3};
		double expected[m] = {10, 11, 12};
		double actual[m];
		matrix_transpose_vector_mult(A, x, actual, m, n);

		for(int i = 0; i < m; ++i)
			ASSERT_EQUAL(actual[i], expected[i]);
	}

	return 0;
}

#define TEST(func, name) do { cout << name << ": " << (func() ? "FAILED\n" : "PASSED\n"); } while(0)

int main()
{
	srand(time(NULL));
	TEST(matrix_mult, "Matrix Math");
	TEST(test_layer, "Normal Layer");
	TEST(sigmoid_layer, "Sigmoid Layer");


	cout << "Tests complete.\n";
	return 0;
}