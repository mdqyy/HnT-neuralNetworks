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


	cout << "Dataset loaded, total elements : " << dataset.getNumSamples() << endl;
	PBDNN pop;
	ifstream inStream(argv[1]);
	inStream >> pop;

	vector<Vec3b> colors = createColorRepartition(pop.getPopulation().size());

	AEMeasurer mae;

	double t = (double) getTickCount();
	t = ((double) getTickCount() - t) / getTickFrequency();
	DiversityMeasurer diversity(pop, dataset, mae);
	vector<NeuralNetworkPtr> population = pop.getPopulation();

	vector<vector<int> > assignedTo = diversity.findBestNetwork();
	vector<vector<FeatureVector> > recomposed = diversity.buildBestOutput();
	vector<int> pngParams = vector<int>();
	pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngParams.push_back(3);
	cout << "Recording Data" << endl;
	for (uint i = 0; i < population.size(); i++) {
		ostringstream dir;
		dir << "network" << i;
		if (mkdir(dir.str().c_str(), S_IRWXU) == 0) {
			for (uint j = 0; j < dataset.getNumSequences(); j++) {
				ostringstream name;
				name << "network" << i << "\/neuralNet" << i << "sample" << j << ".png";
				vector<FeatureVector> features;
				for (uint k = 0; k < dataset[j].size(); k++) {
					population[i]->forward(dataset[j][k]);
					features.push_back(population[i]->getOutputSignal());
				}
				vector<int> color = vector<int>(features.size(), i);
				Mat image = buildColorMapImage(features, 3, color, colors);
				imwrite(name.str(), image, pngParams);
			}
		} else {
			throw invalid_argument("pbdnnCluster : could not create directory");
		}
	}
	ostringstream dirR;
	dirR << "recomposed";
	if (mkdir(dirR.str().c_str(), S_IRWXU) == 0) {
		for (uint j = 0; j < recomposed.size(); j++) {
			ostringstream name;
			name << "recomposed\/recomposedSample" << j << ".png";
			vector<FeatureVector> features;
			Mat image = buildColorMapImage(recomposed[j], 3, assignedTo[j], colors);
			imwrite(name.str(), image, pngParams);
		}
	}
	return EXIT_SUCCESS;
}
