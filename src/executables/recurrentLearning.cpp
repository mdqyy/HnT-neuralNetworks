#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/supervised/RegressionDataset.hpp"
#include "../dataset/unsupervised/UnsupervisedDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/Mask.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../machines/neuralMachines/NeuralNetwork.hpp"
#include "../machines/neuralMachines/layers/InputLayer.hpp"
#include "../machines/neuralMachines/layers/LayerSigmoid.hpp"
#include "../machines/neuralMachines/layers/LayerTanh.hpp"
#include "../machines/neuralMachines/connections/Connection.hpp"
#include "../machines/neuralMachines/PBDNN.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBP.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.hpp"
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	Mat testMat(5, 1, CV_64FC1, 5.0);
	Mat testMat2(5, 1, CV_64FC1, -5.0);
	Mat testMat3(5, 1, CV_64FC1, 0.0);
	Mat meanMat(5, 1, CV_64FC1, 0.0);
	Mat stdevMat(5, 1, CV_64FC1, 1.0);

	FeatureVector testFv(testMat);
	ValueVector mean(meanMat);
	ValueVector stdev(stdevMat);

	LayerPtr il = LayerPtr(new InputLayer(5, mean, stdev));
	LayerPtr th = LayerPtr(new LayerSigmoid(3, "rec",false));
	LayerPtr out = LayerPtr(new LayerSigmoid(2));
	ConnectionPtr c1 = ConnectionPtr(new Connection(il.get(), th.get(),100));
	ConnectionPtr c2 = ConnectionPtr(new Connection(th.get(), out.get(),100));

	vector<LayerPtr> layers;
	layers.push_back(il);
	layers.push_back(th);
	layers.push_back(out);
	vector<ConnectionPtr> connections;
	connections.push_back(c1);
	connections.push_back(c2);
	NeuralNetwork nnTest(layers, connections, "testNN");
	cout << *il << *th;
	cout << *c1;
	cout << *c2;

	cout << nnTest.getOutputSignal()<<endl << endl;

	nnTest.forward(testFv);
	cout << nnTest.getOutputSignal() << endl << endl;

	testFv = FeatureVector(testMat2);
	nnTest.forward(testFv);
	cout << nnTest.getOutputSignal() << endl << endl;

	testFv = FeatureVector(testMat3);
	nnTest.forward(testFv);
	cout << nnTest.getOutputSignal() << endl << endl;
	return EXIT_SUCCESS;
}
