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

void createDatFile(string location, RegressionDataset& dataset) {
	FeatureVector fv;
	ofstream output(location.c_str());
	for (uint i = 0; i < dataset.getNumSequences(); i++) {
		for (uint j = 0; j < dataset[i].size(); j++) {
			fv = dataset[i][j];
			for (uint k = 0; k < fv.getLength(); k++) {
				output << fv[k] << " ";
			}
			output << endl;
		}
	}
	output.close();
}

int main(int argc, char* argv[]) {
	RegressionDataset dataset;
	int numberOfPoints = atoi(argv[1]);
	dataset.setName(argv[2]);
	string saveDatasetLocation = argv[3];
	string saveDatLocation = argv[4];
	realv centerX = -10.0;
	realv centerY = 150.0;
	realv standardDeviationX = 1.0;
	realv standardDeviationY = 1.0;
	smilePotato(&dataset, numberOfPoints, centerX, centerY, standardDeviationX, standardDeviationY);
	smilePotato(&dataset, numberOfPoints, -10.0, -1.0, 5.0, 10.0);
	createDatFile(saveDatLocation, dataset);
	dataset.simpleSave(saveDatasetLocation);
	return EXIT_SUCCESS;
}
