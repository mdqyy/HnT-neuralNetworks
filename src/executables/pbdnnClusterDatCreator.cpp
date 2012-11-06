#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>
#define _POSIX_SOURCE
#include <sys/stat.h>
#include <unistd.h>
#undef _POSIX_SOURCE
#include <stdio.h>
#include <sstream>

#include "../HnT.hpp"


using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	RegressionDataset dataset;
	dataset.load(argv[2]);
	cout << "Validation dataset loaded, total elements : " << dataset.getNumSamples() << endl;

	int iterations = atoi(argv[1]);


	string locationPrefix = argv[3];
	AEMeasurer mae;
	ofstream outStream(argv[4]);
	for(int n = 1;n<=iterations;n++){
		ostringstream file;
		file <<  locationPrefix << n <<".txt";
		PBDNN pop;
		ifstream inStream(file.str().c_str());
		inStream >> pop;
		DiversityMeasurer diversity(pop, dataset, mae);
		vector<realv> bestErrors = diversity.errorsOnBestSample();
		diversity.measurePerformance();
		for(uint i= 0; i<bestErrors.size();i++){
			outStream << bestErrors[i]<<" ";
		}
		outStream<< diversity.getDisagreementScalar()<< " " << endl;
	}
	return EXIT_SUCCESS;
}
