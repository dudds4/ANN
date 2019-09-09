#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

struct Layer;

struct NeuralNetMemory {

	~NeuralNetMemory();

	// delete copy assignment and constructor
	// NeuralNetMemory(const NeuralNetMemory& other) = delete;
	// NeuralNetMemory& operator =(const NeuralNetMemory& other) = delete;

	void allocate(int n_in, int n_out, int n_params);
	void randomizeParams();
	
	double* inputs = nullptr;
	double* activation_cache = nullptr;
	double* outputs = nullptr;
	double* params = nullptr;

private:
	int outSize = 0;
	int paramSize = 0;
};

struct NeuralNet {

	~NeuralNet();
	void addLayer(int n_inputs, int n_outputs, int activationType);
	void initialize();

	void setInputs(const std::vector<double>& v);
	void forward_prop();
	void back_prop();

private:
	std::vector<Layer*> layers;
	bool initialized = false;
	NeuralNetMemory mem;

};

#endif