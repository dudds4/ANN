#include <iostream>
#include <time.h>

#include "neuralnet.h"

using namespace std;

int main()
{
	srand(time(NULL));

	NeuralNet nn;

	nn.addLayer(2, 1, 1);
	nn.initialize();
	nn.setInputs({2, 2});
	
	nn.forward_prop();
	nn.back_prop();

	return 0;
}