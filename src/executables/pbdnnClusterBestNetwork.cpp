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
	arguments.push_back("number of networks");
	arguments.push_back("number of iterations to check");
	arguments.push_back("dat file containing the best performance of each network at every iteration");
	arguments.push_back("location and prefix of files containing the populations during the iterations");
	arguments.push_back("location and file name to save to");
	cout << helper("Create PBDNN from bests", "Create PBDNN using the best networks from a serie of trainings", arguments);
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments" << endl;
		return EXIT_FAILURE;
	}
	ifstream inStream(argv[3]);
	int numberNetworks = atoi(argv[1]);
	int iterations = atoi(argv[2]);
	vector<realv> smallestError = vector<realv>(numberNetworks, 100000.0);
	vector<int> smallestErrorIndex = vector<int>(numberNetworks, 0.0);
	realv error;
	for (int i = 0; i < iterations; i++) {
		for (int j = 0; j < numberNetworks + 1; j++) {
			inStream >> error;
			if (j < numberNetworks) {
				if (error < smallestError[j]) {
					smallestError[j] = error;
					smallestErrorIndex[j] = i+1;
				}
			}
		}
	}
	string locationPrefix = argv[4];
	vector<NeuralNetworkPtr> newPopulation;
	for (int n = 0; n < numberNetworks; n++) {
		cout << smallestErrorIndex[n] << " Best Iteration for " << n << endl;
		ostringstream file;
		file << locationPrefix << smallestErrorIndex[n] << ".txt";
		cout << file.str() << endl;
		PBDNN pop;
		ifstream inStream(file.str().c_str());
		inStream >> pop;
		vector<NeuralNetworkPtr> population = pop.getPopulation();
		newPopulation.push_back(population[n]);
	}
	PBDNN recomposedPopulation(newPopulation);
	ofstream outStream(argv[5]);
	outStream << recomposedPopulation;
	return EXIT_SUCCESS;
}
