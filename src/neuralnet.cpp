#include "neuralnet.h"
#include "layer.h"
#include "stdlib.h"
#include <iostream>

void NeuralNetMemory::allocate(int n_in, int n_out, int n_params)
{
	if(inputs) delete[] inputs;
	inputs = new double[n_in];

	if(activation_cache) delete[] activation_cache;
	activation_cache = new double[n_out];

	if(outputs) delete[] outputs;
	outputs = new double[n_out];

	if(params) delete[] params;
	params = new double[n_params];

	outSize = n_out;
	paramSize = n_params;
}

NeuralNetMemory::~NeuralNetMemory()
{
	if(inputs) delete[] inputs;
	if(activation_cache) delete[] activation_cache;
	if(outputs) delete[] outputs;
	if(params) delete[] params;
}

double randomWeight()
{
	return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
}

void NeuralNetMemory::randomizeParams()
{
	// random initialization of params
	for(int i = 0; i < paramSize; ++i)
		params[i] = randomWeight();	
}

NeuralNet::~NeuralNet()
{
	for(Layer* l : layers)
		delete l;
	
	layers.clear();
}

void NeuralNet::addLayer(int n_inputs, int n_outputs, int activationType)
{
	if(layers.size() && (layers.back()->n_outputs != n_inputs)) 
	{
		throw 1;
	}

	layers.push_back(new SigmoidLayer(n_inputs, n_outputs));
	initialized = false;
}

void NeuralNet::initialize()
{
	if(layers.size() == 0) return;

	int outputMemNeeded = 0;
	int paramMemNeeded = 0;

	for(Layer* l : layers)
	{
		outputMemNeeded += l->n_outputs;

		// W is o x i, b is o x 1
		paramMemNeeded += l->numParams();
	}

	mem.allocate(layers.at(0)->n_inputs, outputMemNeeded, paramMemNeeded);
	mem.randomizeParams();


	initialized = true;
}

void NeuralNet::setInputs(const std::vector<double>& values)
{
	if(!initialized)
		throw 2;
	if(values.size() != layers.at(0)->n_inputs)
		throw 3;

	for(int i = 0; i < values.size(); ++i)
		mem.inputs[i] = values.at(i);
}

void NeuralNet::forward_prop()
{

	const double *i = mem.inputs;
	double* z = mem.activation_cache;
	double* o = mem.outputs;
	double* p = mem.params;

	int num_params;
	for(Layer* l : layers)
	{
		l->forward_prop(i, p, z, o);

		num_params = l->numParams();
		
		i = o;
		p += num_params;
		z += l->n_outputs;
		o += l->n_outputs;
	}
}

void NeuralNet::back_prop()
{
	// int n_out = layers->back().n_outputs;
	// double y[n_out];

	// // TODO: use loss function here
	// for(int i = 0; i < n_out; ++i)
	// 	y[i] = 1;

	// const double *i = mem.inputs;
	// double* o = mem.outputs;
	// double* p = mem.params;
	// double* pd = mem.partials;

	// int num_params;
	// for(Layer* l : layers)
	// {
	// 	l->forward_prop(i, p, pd, o);

	// 	num_params = l->numParams();
		
	// 	i = o;
	// 	p += num_params;
	// 	pd += num_params;
	// 	o += l->n_outputs;
	// }
}