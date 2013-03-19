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
  // ClassificationDataset dataset;
  RegressionDataset dataset;
  dataset.load("../../../../Bases/RIMES-CV/Images-Processed-30px-20px-13/fold1-noOverlapping.xml");
  PBDNN testpop(2,dataset.getFeatureVectorLength(),5,dataset.getMean(), dataset.getStandardDeviation());
  cout << testpop.getPopulation()[0]->getConnections()[0]->getWeights() << endl << endl;
  testpop.regenerate(0);
  cout << testpop.getPopulation()[0]->getConnections()[0]->getWeights() << endl  << endl;
  /*cout << dataset[0][0];
  testpop.forwardSequence(dataset[0]);
  cout << testpop.getOutputSequence()[0];*/
  
  /* Mask mask;
  LearningParams params;
  ofstream log("training.log");
  PopulationBP pbp(testpop,dataset,params,mask,mask,log);

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
  cout << loadedPBDNN.getOutputSequence()[0];*/
  return EXIT_SUCCESS;
}
