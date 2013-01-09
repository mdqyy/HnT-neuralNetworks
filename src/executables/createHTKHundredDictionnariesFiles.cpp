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
#include <map>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("classification dataset");
	arguments.push_back("save folder for the dataset, the folder must contain a lab, labTrain and htk subfolder");
	cout << helper("Create HTK testing files", "Create HTK testing files using a classification dataset.", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
		return EXIT_FAILURE;
	}
	ClassificationDataset datasetBasic;
	datasetBasic.load(argv[1]);
	cout << "loaded dataset" << endl;
	string saveLocation = argv[2];
	vector<FeatureVector> sequence;
	FeatureVector sample;
	AEMeasurer ae;
	for (uint j = 0; j < datasetBasic.getNumSequences(); j++) {
		cout << "Creating dictionnary " << j << "/" << datasetBasic.getNumSequences() << endl;
		ostringstream dictionnaryFileLocation, sequenceFile, wordFile, htkScpFileLocation, labScpFileLocation;
		dictionnaryFileLocation << saveLocation << "dictionnaries/" << j << ".scp";
		htkScpFileLocation << saveLocation << "dataFiles/" << j << ".scp";
		labScpFileLocation << saveLocation << "labFiles/" << j << ".scp";
		sequenceFile << saveLocation << "htk/" << j << ".htk";
		wordFile << saveLocation << "lab/" << j << ".lab";
		ofstream dictionnaryFile(dictionnaryFileLocation.str().c_str());
		ofstream htkScp(htkScpFileLocation.str().c_str());
		ofstream labScp(labScpFileLocation.str().c_str());
		ostringstream word, spacedLetters;
		map<string, string> dictionnary;
		vector<string> target = datasetBasic.getSequenceClasses(j);

		for (uint k = 0; k < target.size(); k++) {
			word << target[k];
			spacedLetters << target[k] << " ";
		}
		dictionnary.insert(pair<string, string>(word.str(), spacedLetters.str()));
		cout << word.str() << endl;
		RNG randomK;
		randomK.next();
		while (dictionnary.size() < 100) {
			ostringstream word2, spacedLetters2;
			uint randK = 0;
			do {
				randomK.next();
				randK = randomK.uniform(0, datasetBasic.getNumSequences());
			} while (randK == j);
			target = datasetBasic.getSequenceClasses(randK);
			for (uint k = 0; k < target.size(); k++) {
				word2 << target[k];
				spacedLetters2 << target[k] << " ";
			}
			dictionnary.insert(pair<string, string>(word2.str(), spacedLetters2.str()));
		}
		map<string, string>::iterator it;
		for (it = dictionnary.begin(); it != dictionnary.end(); it++) {
			dictionnaryFile << (*it).first << " " << (*it).second << endl;
		}
		htkScp << sequenceFile.str() << " " << endl ;
		labScp << wordFile.str() << " " << endl;

	}
	return EXIT_SUCCESS;
}
