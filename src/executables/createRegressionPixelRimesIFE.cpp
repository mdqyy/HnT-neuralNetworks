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

void rimesLoader(string groundTruthFile, string groundTruthFolder, int imageHeight, realv scale, int frameSize, int frameSpace, RegressionDataset* dataset) {
	string line, imageFile, label;
	size_t position;
	ifstream gtFile(groundTruthFile.c_str());
	Mat image;
	ImageFrameExtractor ife(scale,frameSize,frameSpace);
	if (gtFile.is_open()) {
		while (gtFile.good()) {
			getline(gtFile, line);
			position = line.find(" ");
			label = line.substr(position + 1);
			imageFile = groundTruthFolder + line.substr(0, position);
			image = imread(imageFile, 0);
			if (!image.empty() && image.rows == imageHeight) {
			  vector<FeatureVector> frames = ife.getFrames(image);
			  dataset->addSequence(frames, frames);
			}
			else{
			  cout << imageFile <<"is either empty or of the wrong size " << endl; 
			}
		}
		gtFile.close();
	}
}

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("ground truth file containing the location of the datasets");
	arguments.push_back("folder containing the datasets");
	arguments.push_back("normal frame size in pixels");
	arguments.push_back("scaling factor");
	arguments.push_back("frame size");
	arguments.push_back("frame space");
	arguments.push_back("dataset name");
	arguments.push_back("dataset save location");
	arguments.push_back("simple save");
	cout << helper("Create Rimes Regression Dataset", "Create a regression dataset from Rimes files", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc-1 << " given and "<< arguments.size()<<" required" << endl;
		return EXIT_FAILURE;
	}
	RegressionDataset dataset;
	string groundTruthFile = argv[1];
	string groundTruthFolder = argv[2];
	int imageHeight = atoi(argv[3]);
	realv scale = atof(argv[4]);
	int frameSize = atoi(argv[5]);
	int frameSpace = atoi(argv[6]);
	dataset.setName(argv[7]);
	string saveLocation = argv[8];
	int simpleSave = atoi(argv[9]);

	
	rimesLoader(groundTruthFile, groundTruthFolder, imageHeight, scale, frameSize, frameSpace, &dataset);
	if(simpleSave > 0){
	  dataset.simpleSave(saveLocation);
	}
	else{
	  dataset.save(saveLocation);
	}
	return EXIT_SUCCESS;
}
