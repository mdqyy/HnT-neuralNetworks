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

void rimesLoader(string groundTruthFile, string groundTruthFolder, int frameSize, pair<int,int> _frameZone, RegressionDataset* dataset) {
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
			image = imread(imageFile, 0);
			if (!image.empty() && image.rows == 60) {
				vector<FeatureVector> frames = extractFrames(image, frameSize, _frameZone);
				dataset->addSequence(frames, frames);
			}
		}
		gtFile.close();
	}
}

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("ground truth file containing the location of the datasets");
	arguments.push_back("folder containing the datasets");
	arguments.push_back("frame size in pixels");
	arguments.push_back("frame zone start");
	arguments.push_back("frame zone end");
	arguments.push_back("dataset name");
	arguments.push_back("dataset save location");
	cout << helper("Create Rimes Regression Dataset", "Create a regression dataset from Rimes files", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc-1 << " given and "<< arguments.size()<<" required" << endl;
		return EXIT_FAILURE;
	}

	RegressionDataset dataset;
	string groundTruthFile = argv[1];
	string groundTruthFolder = argv[2];
	int frameSize = atoi(argv[3]);
	pair<int,int> frameZone(atoi(argv[4]),atoi(argv[5]));
	dataset.setName(argv[6]);
	string saveLocation = argv[7];
	rimesLoader(groundTruthFile, groundTruthFolder, frameSize, frameZone, &dataset);
	dataset.save(saveLocation);
	return EXIT_SUCCESS;
}
