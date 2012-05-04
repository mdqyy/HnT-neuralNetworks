#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/Mask.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../machines/neuralMachines/NeuralNetwork.hpp"
#include "../machines/neuralMachines/layers/InputLayer.hpp"
#include "../machines/neuralMachines/layers/LayerSigmoid.hpp"
#include "../machines/neuralMachines/layers/LayerTanh.hpp"
#include "../machines/neuralMachines/connections/Connection.hpp"
#include "../machines/neuralMachines/PBDNN.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBP.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBPParams.hpp"

using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  ClassificationDataset dataset;
  dataset.load("../xml/xor.xml");

  /* vector<NeuralNetworkPtr> forwardP=vector<NeuralNetworkPtr>();

  LayerPtr il = LayerPtr(new InputLayer(2,dataset.getMean(),dataset.getStandardDeviation()));
  LayerPtr th = LayerPtr(new LayerSigmoid(2));
  LayerPtr out = LayerPtr(new LayerSigmoid(2));
  ConnectionPtr c1 = ConnectionPtr(new Connection(il.get(),th.get(),20));
  ConnectionPtr c2 = ConnectionPtr(new Connection(th.get(),out.get(),23));

  vector<LayerPtr> layers;
  layers.push_back(il);
  layers.push_back(th);
  layers.push_back(out);
  vector<ConnectionPtr> connections;
  connections.push_back(c1);
  connections.push_back(c2);
  NeuralNetwork nnTest(layers,connections,"testNN");
  

  forwardP.push_back(NeuralNetworkPtr(new NeuralNetwork(nnTest)));
  forwardP.push_back(NeuralNetworkPtr(new NeuralNetwork(nnTest)));
  forwardP.push_back(NeuralNetworkPtr(new NeuralNetwork(nnTest)));

  PBDNN testpop(forwardP);*/
  PBDNN testpop(4,2,2,dataset.getMean(), dataset.getStandardDeviation());
  testpop.forwardSequence(dataset[0]);
  cout << testpop.getOutputSequence()[0];

  Mask mask;
  PopulationBPParams params;
  PopulationBP pbp(testpop,dataset,params,mask,mask);

  pbp.train();
  ClassificationDataset errxor;
  errxor.setName("XorPoped");
  errxor.addClass("positive");
  errxor.addClass("negative");

  for(uint i=0;i<dataset.getNumSequences();i++){
    testpop.forwardSequence(dataset[i]);
    errxor.addSequence(testpop.getOutputSequence()[0],dataset.getSampleClassIndex(i,0));
  }
  errxor.save("../xml/errxor.xml");

  cout << endl <<"Loading and saving" << endl;
  ofstream outStream("testPopSave.txt");
  outStream << testpop;
  cout << "saving done" << endl;
  ifstream in("testPopSave.txt");
  PBDNN loadedPBDNN;
  in >> loadedPBDNN;
  testpop.forwardSequence(dataset[0]);
  loadedPBDNN.forwardSequence(dataset[0]);
  cout << testpop.getOutputSequence()[0];
  cout << loadedPBDNN.getOutputSequence()[0];
  return EXIT_SUCCESS;
}
