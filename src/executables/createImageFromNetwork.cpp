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
  arguments.push_back("population file");
  arguments.push_back("dataset");
  arguments.push_back("frame length");
  cout << helper("Population Image", "Train a population of neural networks on a regression task.Produce images from a population on a given dataset", arguments) << endl;
  if (argc != arguments.size() + 1) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
    return EXIT_FAILURE;
  }
  PBDNN pop;
  LearningParams params;
  ifstream in(argv[1]);
  in >> pop;
  in >> params;
  RegressionDataset dataset;
  dataset.load(argv[2]);
  int frameLength = atoi(argv[3]);
  cout << "Test dataset loaded, total elements : " << dataset.getNumSamples() << endl;
  AEMeasurer mae;
  DiversityMeasurer diversity(pop, dataset, mae);

  cout << "Recording Data" << endl;
  vector<Vec3b> colors = createColorRepartition(pop.getPopulation().size());
  vector<NeuralNetworkPtr> population = pop.getPopulation();
  vector<vector<int> > assignedTo = diversity.findBestNetwork();
  vector<vector<FeatureVector> > recomposed = diversity.buildBestOutput();
  vector<int> pngParams = vector<int>();
  pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
  pngParams.push_back(3);

  for (uint i = 0; i < population.size(); i++) {
    ostringstream dir;
    dir << "network" << i;
    cout << "creating images for network "<< i <<endl;
    if (mkdir(dir.str().c_str(), S_IRWXU) == 0) {
      for (uint j = 0; j < dataset.getNumSequences(); j++) {
	ostringstream name;
	name << "network" << i << "/neuralNet" << i << "sample" << j << ".png";
	vector<FeatureVector> features;
	for (uint k = 0; k < dataset[j].size(); k++) {
	  population[i]->forward(dataset[j][k]);
	  features.push_back(population[i]->getOutputSignal());
	}
	vector<int> color = vector<int>(features.size(), i);
	Mat image = buildColorMapImage(features, frameLength, color, colors);
	imwrite(name.str(), image, pngParams);
      }
    }
    else {
      throw invalid_argument("pbdnnCluster : could not create directory");
    }
  }

  ostringstream dirR;
  dirR << "recomposed";
  if (mkdir(dirR.str().c_str(), S_IRWXU) == 0) {
    for (uint j = 0; j < recomposed.size(); j++) {
      ostringstream name;
      name << "recomposed/recomposedSample" << j << ".png";
      vector<FeatureVector> features;
      Mat image = buildColorMapImage(recomposed[j], frameLength, assignedTo[j], colors);
      imwrite(name.str(), image, pngParams);
    }
  }

  return EXIT_SUCCESS;
}
