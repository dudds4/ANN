#include "layer.h"
#include "neuralnet.h"

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

int test_partial(Layer* l, NeuralNetMemory* mem, int index, int r)
{
	double dp = 0.00000001;
	double out2[l->n_outputs];

	double temp = mem->params[index];
	mem->params[index] += dp;

	l->forward_prop(mem->inputs, mem->params, nullptr, out2);

	double partialMeasured = (out2[r] - mem->outputs[r]) / dp;
	ASSERT_EQUAL(partialMeasured, mem->partials[index]);

	mem->params[index] = temp;

	return 0;
}

int test_general_layer(Layer *l, NeuralNetMemory *mem)
{
	// should fail but not crash
	l->forward_prop(nullptr, mem->params, mem->partials, mem->outputs);
	l->forward_prop(mem->inputs, nullptr, mem->partials, mem->outputs);
	l->forward_prop(mem->inputs, mem->params, mem->partials, nullptr);

	mem->randomizeParams();

	// should run computing partial derivatives
	l->forward_prop(mem->inputs, mem->params, mem->partials, mem->outputs);

	int n_out = l->n_outputs;
	int n_in = l->n_inputs;

	// check the partials for W
	for(int r = 0; r < n_out; ++r)
	{
		for(int c = 0; c < n_in; ++c)
		{
			int index = r*n_out + c;
			if(test_partial(l, mem, index, r))
				return 1;
		}
	}

	// check the partials for b
	for(int r = 0; r < n_out; ++r)
	{
		int index = n_out*n_in + r;
		if(test_partial(l, mem, index, r))
			return 1;		
	}

	return 0;
}

int test_layer()
{
	// a plain layer is just a 
	// matrix multiplication and vector addition

	int n_in = 2, n_out = 2;
	int n_params = n_in*n_out + n_out;

	NeuralNetMemory mem;
	mem.allocate(n_in, n_out, n_params);

	Layer l(n_in,n_out);
	
	// input x
	mem.inputs[0] = -5;
	mem.inputs[1] = -13;

	// W identity matrix
	mem.params[0] = 2;
	mem.params[1] = -7;
	mem.params[2] = 3;
	mem.params[3] = 11;

	// b
	mem.params[4] = -80;
	mem.params[5] = 159;

	// should run without computing partial derivatives
	l.forward_prop(mem.inputs, mem.params, nullptr, mem.outputs);
	ASSERT_VECTOR_EQUALS(mem.outputs, n_out, 1);

	// should run computing partial derivatives
	l.forward_prop(mem.inputs, mem.params, mem.partials, mem.outputs);
	ASSERT_VECTOR_EQUALS(mem.outputs, n_out, 1);

	test_general_layer(&l, &mem);

	return 0;

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
	l.forward_prop(mem.inputs, mem.params, nullptr, mem.outputs);
	ASSERT_VECTOR_EQUALS(mem.outputs, n_out, 0.5);

	mem.params[4] = 100;
	mem.params[5] = -100;	

	// should run computing partial derivatives
	l.forward_prop(mem.inputs, mem.params, mem.partials, mem.outputs);
	ASSERT_EQUAL(mem.outputs[0], 1);
	ASSERT_EQUAL(mem.outputs[1], 0);

	return test_general_layer(&l, &mem);

}

#define TEST(func, name) do { cout << name << ": " << (func() ? "FAILED\n" : "PASSED\n"); } while(0)

int main()
{
	TEST(test_layer, "Normal Layer");
	TEST(sigmoid_layer, "Sigmoid Layer");


	cout << "Tests complete.\n";
	return 0;
}