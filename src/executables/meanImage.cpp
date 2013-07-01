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
  cout << helper("Mean output image", "Visualize the mean outputs of different networks", arguments) << endl;
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
  dataset.simpleLoad(argv[2]);
  int frameLength = atoi(argv[3]);
  cout << "Test dataset loaded, total elements : " << dataset.getNumSamples() << endl;
  AEMeasurer mae = AEMeasurer();
  DiversityMeasurer diversity(pop, dataset, mae);
  cout << "Measuring best outputs and processing mean output" << endl;
  vector<FeatureVector> bests = diversity.getMeanGoodOutput();

  cout << "Recording Data" << endl;
  vector<Vec3b> colors = createColorRepartition(pop.getPopulation().size());
  vector<NeuralNetworkPtr> population = pop.getPopulation();
  vector<int> pngParams = vector<int>();
  pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
  pngParams.push_back(3);

  for (uint i = 0; i < population.size(); i++) {
    ostringstream name;
    name << "network-" << i << "-mean" << ".png";
    vector<FeatureVector> features;
    features.push_back(bests[i]);
    vector<int> color = vector<int>(features.size(), i);
    Mat image = buildColorMapImage(features, frameLength, color, colors);
    imwrite(name.str(), image, pngParams);
  }




  return EXIT_SUCCESS;
}
