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

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/supervised/RegressionDataset.hpp"
#include "../dataset/unsupervised/UnsupervisedDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/Mask.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../machines/neuralMachines/NeuralNetwork.hpp"
#include "../machines/neuralMachines/layers/InputLayer.hpp"
#include "../machines/neuralMachines/layers/LayerSigmoid.hpp"
#include "../machines/neuralMachines/layers/LayerTanh.hpp"
#include "../machines/neuralMachines/connections/Connection.hpp"
#include "../machines/neuralMachines/PBDNN.hpp"
#include "../trainer/errorMeasurers/AEMeasurer.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationClusterBP.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBPParams.hpp"
#include "../utilities/ImageProcessing.hpp"
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	RegressionDataset dataset;
	dataset.load(argv[2]);

	RegressionDataset dataset2;
	dataset2.load(argv[3]);


	cout << "Dataset loaded, total elements : " << dataset.getNumSamples() << endl;
	Mask mask;
	PopulationBPParams params;
	PBDNN pop;
	ifstream inStream(argv[1]);
	inStream >> pop;
	inStream >> params;

	vector<Vec3b> colors = createColorRepartition(pop.getPopulation().size());

	PopulationClusterBP pbp(pop, dataset, params, dataset2,mask, mask);
	AEMeasurer mae;
	DiversityMeasurer diversity(pop, dataset, mae);
	diversity.measurePerformance();
	cout << "Starting diversity" << endl << diversity.getDisagreementMatrix() << endl;
	cout << "Starting overall diversity" << endl << diversity.getDisagreementScalar() << endl;
	double t = (double) getTickCount();
	pbp.train();
	t = ((double) getTickCount() - t) / getTickFrequency();
	cout << "Temps :" << t << endl;
	cout << endl << "Saving network" << endl;
	ofstream outStream("IAMpop.txt");
	outStream << pop;
	DiversityMeasurer diversity2(pop, dataset2, mae);
	vector<NeuralNetworkPtr> population = pop.getPopulation();

	vector<vector<int> > assignedTo = diversity2.findBestNetwork();
	vector<vector<FeatureVector> > recomposed = diversity2.buildBestOutput();
	vector<int> pngParams = vector<int>();
	pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	pngParams.push_back(3);
	cout << "Recording Data" << endl;
	for (uint i = 0; i < population.size(); i++) {
		ostringstream dir;
		dir << "network" << i;
		if (mkdir(dir.str().c_str(), S_IRWXU) == 0) {
			for (uint j = 0; j < dataset2.getNumSequences(); j++) {
				ostringstream name;
				name << "network" << i << "\/neuralNet" << i << "sample" << j << ".png";
				vector<FeatureVector> features;
				for (uint k = 0; k < dataset2[j].size(); k++) {
					population[i]->forward(dataset2[j][k]);
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
