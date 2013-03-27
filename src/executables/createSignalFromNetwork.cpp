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
  arguments.push_back("saveLocation");
  cout << helper("Population signal", "Produce a signal given a database.", arguments) << endl;
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
  string saveLocation(argv[3]);
  cout << "Test dataset loaded, total elements : " << dataset.getNumSamples() << endl;
  AEMeasurer mae;
  DiversityMeasurer diversity(pop, dataset, mae);

  cout << "Recording Data" << endl;
  vector<NeuralNetworkPtr> population = pop.getPopulation();


  for (uint j = 0; j < dataset.getNumSequences(); j++) {
    ostringstream sequenceFile;
    sequenceFile << saveLocation <<  j <<".txt";
    ofstream outputSequence(sequenceFile.str().c_str());
    vector<FeatureVector> features;
    for (uint k = 0; k < dataset[j].size(); k++) {
      FeatureVector sample = dataset[j][k];
      for (uint i = 0; i < population.size(); i++) {
	population[i]->forward(sample);
	outputSequence << mae.totalError(population[i]->getOutputSignal(), sample) << " " ;
      }
      outputSequence << endl;
    }
    outputSequence.close();
  }
  return EXIT_SUCCESS;
}
