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

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("classification dataset");
	arguments.push_back("save folder for the dataset, the folder must contain a lab and htk subfolder");
	cout << helper("Create HTK learning files", "Create HTK learning files using a classification dataset.", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc-1 << " given and "<< arguments.size()<<" required" << endl;
		return EXIT_FAILURE;
	}
	ClassificationDataset datasetBasic;
	datasetBasic.load(argv[1]);
	string saveLocation = argv[2];
	vector<FeatureVector> sequence;
	FeatureVector sample;
	AEMeasurer ae;
	for (uint j = 0; j < datasetBasic.getNumSequences(); j++) {
		ostringstream sequenceFile;
		ostringstream wordFile;
		sequenceFile << saveLocation << "htk/" << j <<".htk";
		wordFile << saveLocation << "lab/"<<j <<".lab";
		sequence = datasetBasic[j];
		ofstream outputSequence(sequenceFile.str().c_str());
		ofstream outputWord(wordFile.str().c_str());
		for (uint k = 0; k < sequence.size(); k++) {
			sample = sequence[k];
			for (uint i = 0; i < sample.getLength(); i++) {
				outputSequence << sample[i] <<" " ;
			}
		outputSequence << endl;
		}
		vector<string> target = datasetBasic.getSequenceClasses(j);
		for(uint k=0; k < target.size();k++){
			outputWord << target[k] << endl;
		}
		outputWord.close();
		outputSequence.close();
	}
	return EXIT_SUCCESS;
}
