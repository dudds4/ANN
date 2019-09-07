#include <iostream>
#include <time.h>

#include "neuralnet.h"

using namespace std;

int main()
{
	srand(time(NULL));

	NeuralNet nn;
	// nn.addLayer(2, 2, 1);
	nn.addLayer(2, 1, 1);
	nn.initialize();
	nn.setInputs({2, 2});
	
	nn.compute();

	return 0;
}