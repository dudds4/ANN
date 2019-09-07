#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

enum ActivationTypes {
	SigmoidActivation,
	TanhActivation,
	ReLUActivation,
	NUM_ACTIVATION_TYPES
};

float sigmoidForward(float x) { 	return 1.0f / (1 + exp(-1.0f * x)); }
float sigmoidPartialDeriv(float x) 
{ 	
	float z = sigmoidForward(x); 
	return z*(1-z);
}

float tanhForward(float x) 
{
	float a = exp(-1.0f*x);
	float b = exp(x);
	return (b - a) / (b + a);
}
float tanhPartialDeriv(float x) 
{
	float a = exp(-1.0f*x);
	float b = exp(x);
	return (b - a) / (b + a);
}

float reluForward(float x) { return x > 0 ? x : 0; }
float reluPartialDeriv(float x) { return x > 0 ? 1 : 0; }

typedef float (*ForwardFunction)(float);
typedef float (*PartialDerivFunction)(float);

ForwardFunction activations[NUM_ACTIVATION_TYPES] = {
	sigmoidForward,
	tanhForward,
	reluForward
};

PartialDerivFunction partialFunctions[NUM_ACTIVATION_TYPES] = {
	sigmoidPartialDeriv,
	tanhPartialDeriv,
	reluPartialDeriv
};

#endif