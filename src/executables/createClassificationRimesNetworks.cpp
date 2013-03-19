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
	cout
			<< helper("Create Population Network Rimes Classification Dataset",
					"Create a classification dataset from Rimes files using the errors produced by passing the frames in a network population.", arguments)
			<< endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
		return EXIT_FAILURE;
	}
	PBDNN pop;
	ifstream inStream(argv[1]);
	inStream >> pop;
	cout << "Loaded population" << endl;
	ClassificationDataset datasetBasic;
	datasetBasic.load(argv[2]);
	cout << "Loaded previous dataset" << endl;
	ClassificationDataset datasetNetworks;
	datasetNetworks.setName(argv[3]);
	cout << "Created new dataset" << endl;
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
			FeatureVector errorSample(population.size() * 4);
			FeatureVector topNetwork(54);
			FeatureVector topSample(54);
			FeatureVector midNetwork(120);
			FeatureVector midSample(120);
			FeatureVector botNetwork(54);
			FeatureVector botSample(54);
			for (uint i = 0; i < population.size(); i++) {
				population[i]->forward(sample);
				networkOutput = population[i]->getOutputSignal();
				errorSample[i] = ae.totalError(networkOutput, sample);

				int v = 0;
				int vtop = 0;
				int vmid = 0;
				int vbot = 0;
				for (int t = 0; t < 3; t++) {
					for (int w = 0; w < 60; w++) {
						if (w < 18) {
							topNetwork[vtop] = networkOutput[v];
							topSample[vtop] = sample[v];
							vtop++;
						}
						if (w >= 10 && w < 50) {
							midNetwork[vmid] = networkOutput[v];
							midSample[vmid] = sample[v];
							vmid++;
						}
						if (w >= 42) {
							botNetwork[vbot] = networkOutput[v];
							botSample[vbot] = sample[v];
							vbot++;
						}
						v++;
					}
				}
				errorSample[i+10] = ae.totalError(topNetwork,topSample);
				errorSample[i+20] = ae.totalError(midNetwork,midSample);
				errorSample[i+30] = ae.totalError(botNetwork,botSample);
			}
			errorSequence.push_back(errorSample);
		}
		datasetNetworks.addSequence(errorSequence, datasetBasic.getSequenceClasses(j));
	}
	cout << "Saving new dataset" << endl;
	datasetNetworks.save(saveLocation);
	return EXIT_SUCCESS;
}
