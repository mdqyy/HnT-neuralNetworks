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
  arguments.push_back("mixed ensemble population");
  arguments.push_back("number of neurons");
  arguments.push_back("output population");
  cout << helper("Mixed ensemble training", "Only trains the last layer, not the complete architecture", arguments) << endl;
  if (argc -1 != arguments.size()) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() <<" required" << endl;
    return EXIT_FAILURE;
  }
  ifstream is(argv[1]);
  MixedEnsembles me;
  is >> me ;
  uint outputNeurons = atoi(argv[2]);
  NeuralNetworkPtr netPtr = me.getOutputNetwork();
  LayerPtr lptr = LayerPtr(new LayerCTC(outputNeurons));
  netPtr->suppressLastLayer();
  netPtr->addLayer(lptr);
  ofstream os(argv[3]);
  os << me;
  return EXIT_SUCCESS;
}
