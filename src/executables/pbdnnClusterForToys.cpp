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
#include <sstream>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	RegressionDataset dataset;
	dataset.load(argv[4]);

	RegressionDataset datasetValid;
	datasetValid.load(argv[5]);

	cout << "dataset loaded, total elements : " << dataset.getNumSamples() << endl;
	int populationSize = atoi(argv[1]);
	int numberOfHiddenUnits = atoi(argv[2]);
	int iterations = atoi(argv[3]);
	AEMeasurer mae;
	PBDNN pop = PBDNN(populationSize, dataset.getFeatureVectorLength(), numberOfHiddenUnits, dataset.getMean(), dataset.getStandardDeviation());
	DiversityMeasurer diversity(pop, datasetValid, mae);
	do {
		pop = PBDNN(populationSize, dataset.getFeatureVectorLength(), numberOfHiddenUnits, dataset.getMean(), dataset.getStandardDeviation());
		diversity.measurePerformance();
		cout << "Disagreement scalar : " << endl << diversity.getDisagreementScalar() << endl;
	} while (diversity.getDisagreementScalar() < 0.007);
	Mask mask;
	LearningParams params;
	params.setMaxIterations(iterations);
	params.setLearningRate(0.01);
	params.setMaxTrainedPercentage(1.0/((realv)(2*populationSize)));
	params.setSavedDuringProcess(true);
	cout << "Training " << endl;
	ofstream log("training.log");
	PopulationClusterBP pbp(pop, dataset, params, datasetValid, mask, mask,log);

	cout << "Starting diversity" << endl << diversity.getDisagreementMatrix() << endl;
	cout << "Starting overall diversity" << endl << diversity.getDisagreementScalar() << endl;
	double t = (double) getTickCount();
	pbp.train();
	t = ((double) getTickCount() - t) / getTickFrequency();
	log << "Temps :" << t << endl;

	vector<NeuralNetworkPtr> population = pop.getPopulation();
	vector<vector<int> > assignedTo = diversity.findBestNetwork();
	cout << "Recording Data" << endl;
	for (uint i = 0; i < population.size(); i++) {
		ostringstream name;
		name << "cluster" << i << ".dat";
		ofstream output(name.str().c_str());
		for (uint j = 0; j < datasetValid.getNumSequences(); j++) {
			vector<FeatureVector> features = datasetValid[j];
			for (uint k = 0; k < features.size(); k++) {
				if(assignedTo[j][k] == i){
					output << features[k][0]<<" "<< features[k][1] <<" " << endl;
				}
			}
		}
		output.close();
	}
	return EXIT_SUCCESS;
}
