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
#include <sstream>

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

int main(int argc, char* argv[]) {
	ClassificationDataset datasetBasic;
	datasetBasic.load(argv[1]);
	string saveLocation = argv[2];
	vector<FeatureVector> sequence, errorSequence;
	FeatureVector sample, networkOutput;
	AEMeasurer ae;
	for (uint j = 0; j < datasetBasic.getNumSequences(); j++) {
		ostringstream sequenceFile;
		ostringstream wordFile;
		sequenceFile << saveLocation << j <<".htk";
		wordFile << saveLocation << j <<".lab";
		sequence = datasetBasic[j];
		ofstream outputSequence(sequenceFile.str().c_str());
		ofstream outputWord(wordFile.str().c_str());
		for (uint k = 0; k < sequence.size(); k++) {
			sample = sequence[k];
			for (uint i = 0; i < sample.getLength(); i++) {
				sequenceFile << sample[i] <<" " ;
			}
			sequenceFile << endl;
		}
		outputWord << datasetBasic.getClassLabel(j)<< " ";
	}
	return EXIT_SUCCESS;
}
