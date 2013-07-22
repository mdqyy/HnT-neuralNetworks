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

int main(int argc, char* argv[]) {
  vector<string> arguments;
  arguments.push_back("mixed ensemble population");
  arguments.push_back("training image dataset");
  arguments.push_back("testing image dataset");
  arguments.push_back("output population");
  arguments.push_back("iterations");
  arguments.push_back("learning rate");
  arguments.push_back("input noise ratio");
  arguments.push_back("max trained percentage");
  arguments.push_back("validation step every N step");
  cout << helper("Mixed ensemble training", "Only trains the last layer, not the complete architecture", arguments) << endl;
  if (argc -1 != arguments.size()) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() <<" required" << endl;
    return EXIT_FAILURE;
  }
  ifstream is(argv[1]);
  string trainingFile(argv[2]);
  string testFile(argv[3]);
  ofstream ofs(argv[4]);
  uint iterations = atoi(argv[5]);
  realv learningRate = atof(argv[6]);
  realv noise = atof(argv[7]);
  realv maxTrainedPC = atof(argv[8]);
  uint validationEveryNIter = atoi(argv[9]);

  MixedEnsembles me;
  is >> me ;
  ImageDataset trainingDataset;
  trainingDataset.load(trainingFile);
  ImageDataset testDataset;
  testDataset.load(testFile);
  LearningParams params;
  params.setActualIteration(0);
  params.setMaxIterations(iterations);
  params.setLearningRate(learningRate);
  params.setMaxTrainedPercentage(maxTrainedPC);
  params.setSavedDuringProcess(true);
  params.setValidateEveryNIteration(validationEveryNIter);
  params.setNoise(noise);
  ofstream log("training.log");
  ImageAutoEncodingME trainer = ImageAutoEncodingME(me,trainingDataset,testDataset,params,log);
  double t = (double) getTickCount();
  trainer.validateIteration();
  trainer.train();
  t = ((double) getTickCount() - t) / getTickFrequency();
  cout << "Time :" << t << endl;
  ofs << me ;
  ofs.close();
  return EXIT_SUCCESS;
}
