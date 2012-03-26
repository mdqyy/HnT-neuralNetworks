#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/Mask.hpp"
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
  dataset.load("errxor.xml");

  InputLayer il(4,dataset.getMean(),dataset.getStandardDeviation());
  LayerSigmoid th(2);
  LayerSigmoid out(2);
  Connection c1(&il,&th,20);
  Connection c2(&th,&out,23);

  vector<Layer*> layers;
  layers.push_back(&il);
  layers.push_back(&th);
  layers.push_back(&out);
  vector<Connection*> connections;
  connections.push_back(&c1);
  connections.push_back(&c2);
  NeuralNetwork nnTest(&il,layers,&out,connections,"testNN");
  cout << il << th;
  cout << c1;
  cout << c2;

  /* Copy test */
  cout << "Copy "<<  endl; 
  NeuralNetwork nnc=*nnTest.clone();

  cout << *nnc.getInputLayer() << endl;
  cout << *nnTest.getInputLayer() << endl;
  cout << *nnc.getInputLayer()->getOutputConnection() << endl;

  Mask mask;
  BackPropParams bpp;
  bpp.setLearningRate(1.0);
  bpp.setLearningRateDecrease(0.95);
  bpp.setMinChangeError(1.0e-09);
  bpp.setMaxIterations(10);
  BackPropagation bp(nnTest,dataset,bpp,mask,mask);
  bp.train();
  
  cout << nnc.getInputLayer();
  cout << *(nnc.getInputLayer()->getOutputConnection());
  cout << *(nnTest.getInputLayer()->getOutputConnection());
  cout << nnc.getInputLayer()->getOutputConnection()->getOutputLayer() << endl ;
  cout << nnTest.getInputLayer()->getOutputConnection()->getOutputLayer();

  nnTest.forward(dataset[0][0]);
  nnc.forward(dataset[0][0]);
  
  cout << nnTest.getOutputSignal() << endl;
  cout << nnc.getOutputSignal() << endl;
  cout << nnc.getOutputLayer()->getInputConnection()->getInputLayer()->getOutputSignal() << endl;
  cout << nnc.getOutputLayer()->getOutputSignal() << endl;
  cout << nnc.getInputLayer()->getOutputSignal() << endl;
  cout << nnc.getOutputLayer()->getInputConnection() << endl;
  cout << nnc.getInputLayer()->getOutputConnection()->getOutputLayer()->getOutputConnection() << endl;

  cout << nnc.getInputLayer()->getOutputConnection() << endl;
  cout << nnc.getOutputLayer()->getInputConnection()->getInputLayer()->getInputConnection() << endl;
    
  return EXIT_SUCCESS;
}
