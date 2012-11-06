#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>

#include "../dataset/supervised/RegressionDataset.hpp"
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
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBP.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.hpp"
#include "../performanceMeasurers/DiversityMeasurer.hpp"

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
