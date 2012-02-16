#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../machines/neuralMachines/NeuralNetwork.hpp"
#include "../machines/neuralMachines/layers/InputLayer.hpp"
#include "../machines/neuralMachines/layers/LayerSigmoid.hpp"
#include "../machines/neuralMachines/layers/LayerTanh.hpp"
#include "../machines/neuralMachines/connections/Connection.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/BackPropagation.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/BackPropParams.hpp"

using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  ClassificationDataset dataset;
  dataset.load("xor.xml");

  InputLayer il(2,dataset.getMean(),dataset.getStandardDeviation());
  LayerSigmoid th(20);
  LayerSigmoid out(2);
  Connection c1(il,th,20);
  Connection c2(th,out,23);

  list<Layer*> layers;
  layers.push_back(&il);
  layers.push_back(&th);
  layers.push_back(&out);
  list<Connection*> connections;
  connections.push_back(&c1);
  connections.push_back(&c2);
  NeuralNetwork nnTest(il,layers,out,connections,"testNN");
  cout << il << th;
  cout << c1;
  cout << c2;

  CrossValidationParams params;

  BackPropParams bpp;
  bpp.setLearningRate(1.0);
  bpp.setLearningRateDecrease(0.95);
  bpp.setMinChangeError(1.0e-09);
  bpp.setMaxIterations(100);
  BackPropagation bp(nnTest,dataset,params,bpp);
  bp.train();

  cout << c1;
  cout << c2;

  
  return EXIT_SUCCESS;
}
