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
#include "../trainer/errorMeasurers/AEMeasurer.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBPBatch.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.hpp"
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	RegressionDataset dataset;
	dataset.load(argv[4]);
	cout << "dataset loaded, total elements : " << dataset.getNumSamples() << endl;
	int populationSize = atoi(argv[1]);
	int numberOfHiddenUnits = atoi(argv[2]);
	int iterations = atoi(argv[3]);
	string filename(argv[5]);
	PBDNN pop(populationSize, dataset.getFeatureVectorLength(), numberOfHiddenUnits, dataset.getMean(), dataset.getStandardDeviation());
	Mask mask;
	LearningParams params;
	params.setMaxIterations(iterations);
	params.setLearningRate(0.1);
	PopulationBPBatch pbp(pop, dataset, params, mask, mask);
	AEMeasurer mae;
	DiversityMeasurer diversity(pop, dataset, mae);
	diversity.measurePerformance();
	cout << "Starting diversity" << endl << diversity.getDisagreementMatrix() << endl;
	cout << "Starting overall diversity" << endl << diversity.getDisagreementScalar() << endl;
	double t = (double) getTickCount();
	pbp.train();
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "Temps :" << t << endl;
	cout << endl << "Saving network" << endl;
	ofstream outStream("IAMpop.txt");
	outStream << pop;

	vector<NeuralNetworkPtr> population = pop.getPopulation();
	RegressionDataset dataset2;
	dataset2.load(argv[6]);
	cout << "Recording Data" << endl;
	for (int i = 0; i < population.size(); i++) {
		ostringstream name, path;
		name << "neuralNet" << i;
		UnsupervisedDataset nnData;
		nnData.setName(name.str());
		for (int j = 0; j < dataset2.getNumSequences(); j++) {
			vector<FeatureVector> features;
			for (uint k = 0; k < dataset2[j].size(); k++) {
				population[i]->forward(dataset2[j][k]);
				features.push_back(population[i]->getOutputSignal());
			}
			nnData.addSequence(features);
		}
		path << filename << name.str() << ".xml";
		nnData.save(path.str());
	}

	return EXIT_SUCCESS;
}
