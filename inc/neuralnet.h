#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

struct Layer;

struct NeuralNetMemory {

	~NeuralNetMemory();

	void allocate(int n_in, int n_out, int n_params);
	void randomizeParams();
	
	double* inputs = nullptr;
	double* outputs = nullptr;
	double* params = nullptr;
	double* partials = nullptr;
private:
	int paramSize = 0;
};

struct NeuralNet {

	~NeuralNet();
	void addLayer(int n_inputs, int n_outputs, int activationType);
	void initialize();

	void setInputs(const std::vector<double>& v);
	void compute();

private:
	std::vector<Layer*> layers;
	bool initialized = false;
	NeuralNetMemory mem;
};

#endif