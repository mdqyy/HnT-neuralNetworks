#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>
#define _POSIX_SOURCE
#include <sys/stat.h>
#include <unistd.h>
#undef _POSIX_SOURCE
#include <stdio.h>

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
#include "../trainer/errorMeasurers/AEMeasurer.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationClusterBP.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.hpp"
#include "../utilities/ImageProcessing.hpp"
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	ifstream inStream(argv[3]);
	int numberNetworks = atoi(argv[1]);
	int iterations = atoi(argv[2]);
	vector<realv> smallestError = vector<realv>(numberNetworks, 100000.0);
	vector<int> smallestErrorIndex = vector<int>(numberNetworks, 0.0);
	realv error;
	for (int i = 0; i < iterations; i++) {
		for (int j = 0; j < numberNetworks + 1; j++) {
			inStream >> error;
			if (j < numberNetworks) {
				if (error < smallestError[j]) {
					smallestError[j] = error;
					smallestErrorIndex[j] = i+1;
				}
			}
		}
	}
	string locationPrefix = argv[4];
	vector<NeuralNetworkPtr> newPopulation;
	for (int n = 0; n < numberNetworks; n++) {
		cout << smallestErrorIndex[n] << " Best Iteration for " << n << endl;
		ostringstream file;
		file << locationPrefix << smallestErrorIndex[n] << ".txt";
		cout << file.str() << endl;
		PBDNN pop;
		ifstream inStream(file.str().c_str());
		inStream >> pop;
		vector<NeuralNetworkPtr> population = pop.getPopulation();
		newPopulation.push_back(population[n]);
	}
	PBDNN recomposedPopulation(newPopulation);
	ofstream outStream(argv[5]);
	outStream << recomposedPopulation;
	return EXIT_SUCCESS;
}
