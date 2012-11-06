#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>

#include "../HnT.hpp"


using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  ClassificationDataset dataset;
  dataset.load("../xml/xor.xml");

  LayerPtr il = LayerPtr(new InputLayer(2,dataset.getMean(),dataset.getStandardDeviation()));
  LayerPtr th = LayerPtr(new LayerSigmoid(3));
  LayerPtr out = LayerPtr(new LayerSigmoid(2));
  ConnectionPtr c1 = ConnectionPtr(new Connection(il.get(),th.get(),500));
  ConnectionPtr c2 = ConnectionPtr(new Connection(th.get(),out.get(),700));

  vector<LayerPtr> layers;
  layers.push_back(il);
  layers.push_back(th);
  layers.push_back(out);
  vector<ConnectionPtr> connections;
  connections.push_back(c1);
  connections.push_back(c2);
  NeuralNetwork nnTest(layers,connections,"testNN");
  cout << *il << *th;
  cout << *c1;
  cout << *c2;
  nnTest.forward(dataset[0][0]);
  cout << nnTest.getOutputSignal() << dataset.getTargetSample(0,0) <<endl;
  nnTest.forward(dataset[1][0]);
  cout << nnTest.getOutputSignal() << dataset.getTargetSample(1,0) << endl;


  /* Copy test */
  /* cout << "Copy "<<  endl; 
  NeuralNetworkPtr nnc=NeuralNetworkPtr(nnTest.clone());

  cout << *nnc->getInputLayer() << endl;
  cout << *nnTest.getInputLayer() << endl;
  cout << *nnc->getInputLayer()->getOutputConnection() << endl;
  */
  cout << "Training" << endl;
  Mask mask;
  LearningParams bpp;
  bpp.setLearningRate(0.5);
  bpp.setLearningRateDecrease(0.95);
  bpp.setMinChangeError(1.0e-9);
  bpp.setMaxIterations(1000);
  bpp.setTask(BP_CLASSIFICATION);
  BackPropagation bp(nnTest,dataset,bpp,mask,mask);
  bp.train();
  cout << "Ended training" << endl;
  nnTest.forward(dataset[0][0]);
  cout << nnTest.getOutputSignal() << endl;
  nnTest.forward(dataset[1][0]);
  cout << nnTest.getOutputSignal() << endl;

 cout << *il << *th;
  cout << *c1;
  cout << *c2;
  /*
  cout << nnc->getInputLayer();
  cout << *(nnc->getInputLayer()->getOutputConnection());
  cout << *(nnTest.getInputLayer()->getOutputConnection());
  cout << nnc->getInputLayer()->getOutputConnection()->getOutputLayer() << endl ;
  cout << nnTest.getInputLayer()->getOutputConnection()->getOutputLayer();

  nnTest.forward(dataset[0][0]);
  nnc->forward(dataset[0][0]);
  
  cout << nnTest.getOutputSignal() << endl;
  cout << nnc->getOutputSignal() << endl;
  cout << nnc->getOutputLayer()->getInputConnection()->getInputLayer()->getOutputSignal() << endl;
  cout << nnc->getOutputLayer()->getOutputSignal() << endl;
  cout << nnc->getInputLayer()->getOutputSignal() << endl;
  cout << *nnc->getOutputLayer()->getInputConnection() << endl;
  cout << *nnc->getInputLayer()->getOutputConnection()->getOutputLayer()->getOutputConnection() << endl;

  cout << *nnc->getInputLayer()->getOutputConnection() << endl;
  cout << *nnc->getOutputLayer()->getInputConnection()->getInputLayer()->getInputConnection() << endl;
    

  cout << endl <<"Loading and saving" << endl;
  ofstream outStream("testNNSave.txt");
  outStream << nnTest;
  cout << "saving done" << endl;
  ifstream in("testNNSave.txt");
  NeuralNetwork loadedNN;
  in >> loadedNN;
  nnTest.forward(dataset[0][0]);
  loadedNN.forward(dataset[0][0]);
  cout << nnTest.getOutputSignal() << endl;
  cout << loadedNN.getOutputSignal() << endl;*/
  return EXIT_SUCCESS;
}
