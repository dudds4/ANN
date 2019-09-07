#ifndef LAYER_H
#define LAYER_H

struct Layer {

	Layer() = delete;
	Layer(int i, int o);

	int numParams() const;
	void forward_prop(const double *x, const double *params, double *partials, double *a);
	// virtual void activationDeriv(double *z, double *d, int n)=0;
	// virtual double back_prop();

// private:
	int n_inputs, n_outputs;

	virtual void activation(double *z, double *a, double *d, int n);
	// double *x=nullptr, *W=nullptr, *b=nullptr;
};

struct SigmoidLayer : Layer 
{
	SigmoidLayer() = delete;
	SigmoidLayer(int i, int o) : Layer(i, o) {}

	virtual void activation(double *z, double *a, double *d, int n);

	// virtual void activationDeriv(double *z, double *d, int n)
	// {
	// 	//TODO: optimize this loop
	// 	for(int i = 0; i < n; ++i)
	// 	{
	// 		double s = sigmoid(z[i]);
	// 		d[i] = s*(1-s);
	// 	}
	// }
};

#endif