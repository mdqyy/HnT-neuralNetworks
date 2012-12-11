#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <stdio.h>
#include <fstream>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("population");
	arguments.push_back("classification dataset");
	arguments.push_back("output classification dataset");
	arguments.push_back("output classification dataset save location");
	cout << helper("Create Population Network Rimes Classification Dataset", "Create a classification dataset from Rimes files using the errors produced by passing the frames in a network population.", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc-1 << " given and "<< arguments.size()<<" required" << endl;
		return EXIT_FAILURE;
	}
	PBDNN pop;
	ifstream inStream(argv[1]);
	inStream >> pop;
	ClassificationDataset datasetBasic;
	datasetBasic.load(argv[2]);
	ClassificationDataset datasetNetworks;
	datasetNetworks.setName(argv[3]);
	addDictionaryClasses(&datasetNetworks);
	string saveLocation = argv[4];
	vector<NeuralNetworkPtr> population = pop.getPopulation();
	vector<FeatureVector> sequence;
	FeatureVector sample, networkOutput;
	AEMeasurer ae;
	cout << "Loaded datasets and population, now processing : " << endl;
	for (uint j = 0; j < datasetBasic.getNumSequences(); j++) {
		sequence = datasetBasic[j];
		vector<FeatureVector> errorSequence;
		for (uint k = 0; k < sequence.size(); k++) {
			sample = sequence[k];
			FeatureVector errorSample(population.size());
			for (uint i = 0; i < population.size(); i++) {
				population[i]->forward(sample);
				networkOutput = population[i]->getOutputSignal();
				errorSample[i] = ae.totalError(networkOutput,sample);
			}
			errorSequence.push_back(errorSample);
		}
		datasetNetworks.addSequence(errorSequence, datasetBasic.getSequenceClasses(j));
	}
	cout << "Saving new dataset" << endl;
	datasetNetworks.save(saveLocation);
	return EXIT_SUCCESS;
}
