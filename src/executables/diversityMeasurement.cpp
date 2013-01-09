#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("database");
	arguments.push_back("population file");
	cout << helper("Measure diversity", "Measure diversity of a population on a validation dataset.", arguments);
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments" << endl;
		return EXIT_FAILURE;
	}
	RegressionDataset dataset;
	dataset.load(argv[1]);
	PBDNN pop;
	ifstream inStream(argv[2]);
	inStream >> pop;
	AEMeasurer mae;
	DiversityMeasurer diversity(pop, dataset, mae);
	diversity.measurePerformance();
	cout << diversity.getDisagreementMatrix() << endl;
	cout << diversity.getDisagreementScalar() << endl;
	return EXIT_SUCCESS;
}
