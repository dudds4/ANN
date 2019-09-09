#ifndef LAYER_H
#define LAYER_H

struct NeuralNetMemory;

struct Layer {

	Layer() = delete;
	Layer(int i, int o);

	int numParams() const;
	void forward_prop(const NeuralNetMemory* mem);
	void forward_prop(const double *x, const double *params, double *z, double *a);
	void back_prop(const double* da, NeuralNetMemory* mem, double *dparams, double *dx);
	// virtual void activationDeriv(double *z, double *d, int n)=0;
	// virtual double back_prop();

// private:
	int n_inputs, n_outputs;

	virtual void activation(double *z, double *a);
	virtual void activationDeriv(const double *z, double *dz);
	// double *x=nullptr, *W=nullptr, *b=nullptr;
};

struct SigmoidLayer : Layer 
{
	SigmoidLayer() = delete;
	SigmoidLayer(int i, int o) : Layer(i, o) {}

	virtual void activation(double *z, double *a);
	virtual void activationDeriv(const double *z, double *dz);

};

#endif