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
	arguments.push_back("regression dataset");
	arguments.push_back("save folder location");
	arguments.push_back("number of image rows in a data vector");
	cout << helper("Dataset to images", "Create images from a regression dataset.", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
		return EXIT_FAILURE;
	}
	RegressionDataset dataset;
	dataset.load(argv[1]);
	uint rowsPerVector = atoi(argv[3]);
	vector<int> params = vector<int>();
	params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	params.push_back(3);
	for (uint i = 0; i < dataset.getNumSequences(); i++) {
		ostringstream name;
		name << argv[2] << i << dataset.getName() << ".png";
		vector<FeatureVector> sequence = dataset[i];
		uint imageLength = sequence.size() * rowsPerVector;
		uint vectorLengthPerRow = sequence[0].getLength() / rowsPerVector;
		Mat image(vectorLengthPerRow, imageLength, CV_8UC1, Scalar(0));
		uint imageCol = 0;
		uint vectorIndex = 0;
		for (uint j = 0; j < sequence.size(); j++) {
			for (uint k = 0; k < rowsPerVector; k++) {
				for (uint l = 0; l < vectorLengthPerRow; l++) {
					image.at<uchar>(l, imageCol) = (uchar) (sequence[j][vectorIndex] * 255);
					vectorIndex++;
				}
				imageCol++;
			}
			vectorIndex = 0;
		}
		imwrite(name.str(), image, params);
	}
	return EXIT_SUCCESS;
}
