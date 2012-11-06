#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>

#include "../dataset/supervised/ClassificationDataset.hpp"
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
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBP.hpp"
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.hpp"

using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
   ClassificationDataset dataset;
  //  RegressionDataset dataset;
  dataset.load("../xml/xor.xml");
  PBDNN testpop(1,dataset.getFeatureVectorLength(),5,dataset.getMean(), dataset.getStandardDeviation());
  cout << dataset[0][0];
  testpop.forwardSequence(dataset[0]);
  cout << testpop.getOutputSequence()[0];
  
  Mask mask;
  LearningParams params;
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
