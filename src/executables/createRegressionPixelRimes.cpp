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

using namespace std;
using namespace cv;

void rimesLoader(string groundTruthFile, string groundTruthFolder, int frameSize, RegressionDataset* dataset) {
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
				vector<FeatureVector> frames = extractFrames(image, frameSize);
				dataset->addSequence(frames, frames);
			}
		}
		gtFile.close();
	}
}

int main(int argc, char* argv[]) {
	RegressionDataset dataset;
	string groundTruthFile = argv[1];
	string groundTruthFolder = argv[2];
	int frameSize = atoi(argv[3]);
	dataset.setName(argv[4]);
	string saveLocation = argv[5];
	rimesLoader(groundTruthFile, groundTruthFolder, frameSize, &dataset);
	dataset.save(saveLocation);
	return EXIT_SUCCESS;
}
