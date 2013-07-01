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
  cout << helper("Connector test", "Connector test", arguments) << endl;
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
  cout << "Test dataset loaded, total elements : " << dataset.getNumSamples() << endl;
  vector<NeuralNetworkPtr> population = pop.getPopulation();
  vector<LayerPtr> hiddenLayers = vector<LayerPtr>();
  cout << "loaded" << endl;
  for(uint i = 0; i < population.size(); i++) {
    int prior = population[i]->getHiddenLayers().size();
    population[i]->forward(dataset[0][0]);
    population[i]->suppressLastLayer();
    population[i]->forward(dataset[0][0]);
    hiddenLayers.push_back(population[i]->getHiddenLayers().back());
    cout << "net "<< i <<"  "<<prior<< " "<< population[i]->getHiddenLayers().size()<< endl;
  }
  for(uint i = 0; i < population.size(); i++) {
    population[0]->forward(dataset[0][0]);
  }

  cout << "forwarded" << endl;
  Connector nokia(hiddenLayers);
  FeatureVector out = nokia.concatenateOutputs();
  cout <<"Length "<<out.getLength() << endl;
  cout << "First " << out[0] << endl;
  cout << "Second " << out[1] << endl;
  cout << hiddenLayers[0]->getOutputSignal()[0] << endl;
  cout << population[0]->getHiddenLayers()[1]->getOutputSignal()[0] << endl;
  cout << population[0]->getOutputSignal()[0] << endl;
  cout << hiddenLayers[0]->getOutputSignal()[1] << endl;
  cout << population[0]->getHiddenLayers()[1]->getOutputSignal()[1] << endl;
  cout << population[0]->getOutputSignal()[1] << endl;
  cout << hiddenLayers.back()->getOutputSignal()[1] << endl;
  cout << population.back()->getHiddenLayers()[1]->getOutputSignal()[1] << endl;
  cout << population.back()->getOutputSignal()[1] << endl;
  
  return EXIT_SUCCESS;
}
