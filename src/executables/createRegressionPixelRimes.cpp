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

#include "../dataset/supervised/RegressionDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../utilities/ImageProcessing.hpp"
#include "../utilities/TextUtilities.hpp"

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
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBPParams.hpp"
#include "../utilities/ImageProcessing.hpp"

using namespace std;
using namespace cv;

void addDictionaryClasses(ClassificationDataset* dataset){
	dataset->addClass("a");
	dataset->addClass("â");
	dataset->addClass("à");
	dataset->addClass("b");
	dataset->addClass("c");
	dataset->addClass("d");
	dataset->addClass("e");
	dataset->addClass("ê");
	dataset->addClass("é");
	dataset->addClass("è");
	dataset->addClass("f");
	dataset->addClass("g");
	dataset->addClass("h");
	dataset->addClass("i");
	dataset->addClass("j");
	dataset->addClass("k");
	dataset->addClass("l");
	dataset->addClass("m");
	dataset->addClass("n");
	dataset->addClass("o");
	dataset->addClass("p");
	dataset->addClass("q");
	dataset->addClass("r");
	dataset->addClass("s");
	dataset->addClass("t");
	dataset->addClass("u");
	dataset->addClass("ù");
	dataset->addClass("ü");
	dataset->addClass("v");
	dataset->addClass("w");
	dataset->addClass("x");
	dataset->addClass("y");
	dataset->addClass("z");
	dataset->addClass("A");
	dataset->addClass("B");
	dataset->addClass("C");
	dataset->addClass("D");
	dataset->addClass("E");
	dataset->addClass("F");
	dataset->addClass("G");
	dataset->addClass("H");
	dataset->addClass("I");
	dataset->addClass("J");
	dataset->addClass("K");
	dataset->addClass("L");
	dataset->addClass("M");
	dataset->addClass("N");
	dataset->addClass("O");
	dataset->addClass("P");
	dataset->addClass("Q");
	dataset->addClass("R");
	dataset->addClass("S");
	dataset->addClass("T");
	dataset->addClass("U");
	dataset->addClass("V");
	dataset->addClass("W");
	dataset->addClass("X");
	dataset->addClass("Y");
	dataset->addClass("Z");
	dataset->addClass("'");
	dataset->addClass("°");
	dataset->addClass("%");
	dataset->addClass("-");
	dataset->addClass("/");
	dataset->addClass("0");
	dataset->addClass("1");
	dataset->addClass("2");
	dataset->addClass("3");
	dataset->addClass("4");
	dataset->addClass("5");
	dataset->addClass("6");
	dataset->addClass("7");
	dataset->addClass("8");
	dataset->addClass("9");
}

int main(int argc, char* argv[]) {
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
	vector<FeatureVector> sequence, errorSequence;
	FeatureVector sample, networkOutput;
	AEMeasurer ae;
	for (int j = 0; j < datasetBasic.getNumSequences(); j++) {
		sequence = datasetBasic[j];
		errorSequence = vector<FeatureVector>(sequence.size()),FeatureVector(population.size());
		for (int k = 0; k < sequence.size(); k++) {
			sample = sequence[k];
			for (int i = 0; i < population.size(); i++) {
				population[i]->forward(sample);
				networkOutput = population[i]->getOutputSignal();
				errorSequence[k][i] = ae.totalError(networkOutput,sample);
			}
		}
		datasetNetworks.addSequence(errorSequence, datasetBasic.getSequenceClasses(j));
	}
	datasetNetworks.save(saveLocation);
	return EXIT_SUCCESS;
}
