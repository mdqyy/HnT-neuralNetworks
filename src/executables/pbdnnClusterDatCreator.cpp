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

	vector<string> arguments;
	arguments.push_back("number of iterations");
	arguments.push_back("validation dataset");
	arguments.push_back("location and prefix of the learning files");
	arguments.push_back("dat file name and location");
	cout << helper("Population error and diversity extractor", " Creates a dat file containing ", arguments) << endl << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments" << endl;
		return EXIT_FAILURE;
	}
	RegressionDataset dataset;
	dataset.load(argv[2]);
	cout << "Validation dataset loaded, total elements : " << dataset.getNumSamples() << endl;

	int iterations = atoi(argv[1]);

	string locationPrefix = argv[3];
	AEMeasurer mae;
	ofstream outStream(argv[4]);
	for (int n = 1; n <= iterations; n++) {
		double t = (double) getTickCount();

		ostringstream file;
		file << locationPrefix << n << ".txt";
		PBDNN pop;
		ifstream inStream(file.str().c_str());
		inStream >> pop;
		DiversityMeasurer diversity(pop, dataset, mae);
		vector<realv> bestErrors = diversity.errorsOnBestSample();
		//		diversity.measurePerformance();
		for (uint i = 0; i < bestErrors.size(); i++) {
			outStream << bestErrors[i] << " ";
		}
		t = ((double) getTickCount() - t) / getTickFrequency();
		cout << "Time :" << t << endl;
		/*	outStream << diversity.getDisagreementScalar() << " " << endl;*/
	}
	return EXIT_SUCCESS;
}
