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
  arguments.push_back("input height");
  arguments.push_back("output location");
  cout << helper("Weigh images", "Visualize the weights of the encoding filter of different networks", arguments) << endl;
  if (argc != arguments.size() + 1) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
    return EXIT_FAILURE;
  }
  PBDNN pop;
  ifstream in(argv[1]);
  in >> pop;
  string location = argv[3];
  int inputHeight = atoi(argv[2]);
  cout << "Recording Data" << endl;
  vector<NeuralNetworkPtr> population = pop.getPopulation();
  vector<int> pngParams = vector<int>();
  pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
  pngParams.push_back(3);

  for (uint i = 0; i < population.size(); i++) {
    ostringstream weightFolder;
    weightFolder << location << "/weightsNet-"<<i;
    if (mkdir(weightFolder.str().c_str(), S_IRWXU) == 0){
      cout << "created directory for net" << i << endl;
    }
    for(uint j = 0; j < population[i]->getConnections()[0]->getWeights().rows;j++){
      ostringstream name;
      name << weightFolder.str() << "/weights"<<j << ".png";
      Mat image = createWeightImage(population[i]->getConnections()[0]->getWeights(),inputHeight,j);
      imwrite(name.str(), image, pngParams);
    }
  }
  return EXIT_SUCCESS;
}
