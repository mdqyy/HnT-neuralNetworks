#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>

#include "../HnT.hpp"


using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  RegressionDataset dataset;
  dataset.load(argv[1]);
  PBDNN pop(5,dataset.getFeatureVectorLength(),5,dataset.getMean(), dataset.getStandardDeviation());
  AEMeasurer mae;
  DiversityMeasurer diversity(pop, dataset, mae);
  diversity.measurePerformance();
  cout << diversity.getDisagreementMatrix() << endl;
  return EXIT_SUCCESS;
}
