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
	ClassificationDataset datasetBasic;
	datasetBasic.load(argv[1]);
	string saveLocation = argv[2];
	vector<FeatureVector> sequence;
	FeatureVector sample;
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
