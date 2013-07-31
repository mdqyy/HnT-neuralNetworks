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
#include "../dataset/SequenceClassDataset.hpp"

using namespace std;
using namespace cv;


void rimesLoader(string groundTruthFile, string groundTruthFolder, int frameSize, SequenceClassDataset* dataset) {
	string line, imageFile, label;
	size_t position;
	ifstream gtFile(groundTruthFile.c_str());
	Mat image;

	if (gtFile.is_open()) {
		while (gtFile.good()) {
			getline(gtFile, line);
			position = line.find(" ");
			label = line.substr(position + 1);
			imageFile = groundTruthFolder + line.substr(0, position);
			label = rtrim(label);
			image = imread(imageFile, 0);
			if (!image.empty()) {
				vector<string> labelSequence = extractLabelSequence(label);
				dataset->addSequence(labelSequence);
			}
			else {
				cout << "Is empty label : " << label << endl;
				cout << "In file : "<< imageFile << endl;
			}
		}
		gtFile.close();
	}
}

int main(int argc, char* argv[]) {
	std::setlocale(LC_ALL, "" );
	vector<string> arguments;
	arguments.push_back("ground truth file containing the location of the datasets");
	arguments.push_back("folder containing the datasets");
	arguments.push_back("frame size in pixels");
	arguments.push_back("dataset name");
	arguments.push_back("dataset save location");
	cout << helper("Create Rimes Classification Dataset", "Create a classification dataset from Rimes files", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
		return EXIT_FAILURE;
	}
	SequenceClassDataset dataset;
	addDictionaryClasses(&dataset);
	string groundTruthFile = argv[1];
	string groundTruthFolder = argv[2];
	int frameSize = atoi(argv[3]);
	string saveLocation = argv[5];
	rimesLoader(groundTruthFile, groundTruthFolder, frameSize, &dataset);
	dataset.save(saveLocation);
	return EXIT_SUCCESS;
}
