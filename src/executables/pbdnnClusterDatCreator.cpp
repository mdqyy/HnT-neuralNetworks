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

	RegressionDataset dataset;
	dataset.load(argv[2]);
	cout << "Validation dataset loaded, total elements : " << dataset.getNumSamples() << endl;

	int iterations = atoi(argv[1]);


	string locationPrefix = argv[3];
	AEMeasurer mae;
	ofstream outStream(argv[4]);
	for(int n = 1;n<=iterations;n++){
		ostringstream file;
		file <<  locationPrefix << n <<".txt";
		PBDNN pop;
		ifstream inStream(file.str().c_str());
		inStream >> pop;
		DiversityMeasurer diversity(pop, dataset, mae);
		vector<realv> bestErrors = diversity.errorsOnBestSample();
		diversity.measurePerformance();
		for(uint i= 0; i<bestErrors.size();i++){
			outStream << bestErrors[i]<<" ";
		}
		outStream<< diversity.getDisagreementScalar()<< " " << endl;
	}
	return EXIT_SUCCESS;
}
