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

using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  ClassificationDataset dataset;
  dataset.load("xor.xml");

  vector<NeuralNetwork*> forwardP=vector<NeuralNetwork*>();

  InputLayer il(2,dataset.getMean(),dataset.getStandardDeviation());
  LayerSigmoid th(3);
  LayerSigmoid out(2);
  Connection c1(il,th,520);
  Connection c2(th,out,523);
  list<Layer*> layers;
  layers.push_back(&il);
  layers.push_back(&th);
  layers.push_back(&out);
  list<Connection*> connections;
  connections.push_back(&c1);
  connections.push_back(&c2);
  NeuralNetwork nnTest(il,layers,out,connections,"f1");
  forwardP.push_back(&nnTest);

  InputLayer il1(2,dataset.getMean(),dataset.getStandardDeviation());
  LayerSigmoid th1(3);
  LayerSigmoid out1(2);
  Connection c11(il1,th1,420);
  Connection c12(th1,out1,423);
  list<Layer*> layers1;
  layers1.push_back(&il1);
  layers1.push_back(&th1);
  layers1.push_back(&out1);
  list<Connection*> connections1;
  connections1.push_back(&c11);
  connections1.push_back(&c12);
  NeuralNetwork nnTest1(il1,layers1,out1,connections1,"f2");
  forwardP.push_back(&nnTest1);

  InputLayer il2(2,dataset.getMean(),dataset.getStandardDeviation());
  LayerSigmoid th2(3);
  LayerSigmoid out2(2);
  Connection c21(il2,th2,201);
  Connection c22(th2,out2,232);
  list<Layer*> layers2;
  layers2.push_back(&il2);
  layers2.push_back(&th2);
  layers2.push_back(&out2);
  list<Connection*> connections2;
  connections2.push_back(&c21);
  connections2.push_back(&c22);
  NeuralNetwork nnTest2(il2,layers2,out2,connections2,"f3");
  forwardP.push_back(&nnTest2);

  InputLayer il3(2,dataset.getMean(),dataset.getStandardDeviation());
  LayerSigmoid th3(3);
  LayerSigmoid out3(2);
  Connection c31(il3,th3,205);
  Connection c32(th3,out3,231);
  list<Layer*> layers3;
  layers3.push_back(&il3);
  layers3.push_back(&th3);
  layers3.push_back(&out3);
  list<Connection*> connections3;
  connections3.push_back(&c31);
  connections3.push_back(&c32);
  NeuralNetwork nnTest3(il3,layers3,out3,connections3,"f4");
  forwardP.push_back(&nnTest3);

  PBDNN testpop(forwardP);
  testpop.forwardSequence(dataset[0]);
  cout << testpop.getOutputSequence()[0];

  Mask mask;
  PopulationBP pbp(testpop,dataset,mask,mask);
  pbp.train();
  
  ClassificationDataset errxor;
  errxor.setName("XorPoped");
  errxor.addClass("positive");
  errxor.addClass("negative");

  for(uint i=0;i<dataset.getNumSequences();i++){
    testpop.forwardSequence(dataset[i]);
    errxor.addSequence(testpop.getOutputSequence()[0],dataset.getSampleClassIndex(i,0));
  }
  errxor.save("errxor.xml");
  return EXIT_SUCCESS;
}
