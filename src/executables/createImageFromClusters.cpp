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

vector<uint> createLabels(char* file){
  string line;
  vector<uint> labels = vector<uint>();
  ifstream labFile(file);
  if(labFile.is_open()){
    while(labFile.good()){
      getline(labFile,line);
      labels.push_back(atoi(line.c_str()));
    }
  }
  return labels;
}

int main(int argc, char* argv[]) {
  vector<string> arguments;
  arguments.push_back("dataset");
  arguments.push_back("number of clusters");
  arguments.push_back("cluster file");
  arguments.push_back("frame length");
  cout << helper("Cluster Image", "Get batch of images corresponding to ", arguments) << endl;
  if (argc != arguments.size() + 1) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
    return EXIT_FAILURE;
  }
  RegressionDataset dataset;
  dataset.simpleLoad(argv[1]);
  uint numClusters = atoi(argv[2]);
  vector<uint> labels = createLabels(argv[3]);
  int frameLength = atoi(argv[4]);
  cout << "Dataset loaded, total elements : " << dataset.getNumSamples() << endl;
  cout << "Labels loaded, total elements : " << labels.size() << endl;
  
  cout << "Recording Data" << endl;
  vector<int> pngParams = vector<int>();
  pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
  pngParams.push_back(3);

  for (uint i = 0; i < numClusters; i++) {
    ostringstream dir;
    dir << "cluster" << i;
    if (mkdir(dir.str().c_str(), S_IRWXU) == 0) {
    }
  }

  for (uint j = 0; j < dataset.getNumSequences(); j++) {
    ostringstream name;
    name << "cluster" << labels[j] << "/sample" << j << ".png";
    vector<FeatureVector> features = dataset[j];
    Mat image = buildImage(features, frameLength);
    imwrite(name.str(), image, pngParams);
  }

  return EXIT_SUCCESS;
}
