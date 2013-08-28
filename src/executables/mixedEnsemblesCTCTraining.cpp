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
#include "../trainer/supervisedTrainers/neuralNetworkTrainers/MECTCTrainer.hpp"
#include "../dataset/SequenceClassDataset.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  vector<string> arguments;
  arguments.push_back("mixed ensemble population");
  arguments.push_back("training image dataset");
  arguments.push_back("training class dataset");
  arguments.push_back("testing image dataset");
  arguments.push_back("testing class dataset");
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
  string trainingClassFile(argv[3]);
  string testFile(argv[4]);
  string testClassFile(argv[5]);
  ofstream ofs(argv[6]);
  uint iterations = atoi(argv[7]);
  realv learningRate = atof(argv[8]);
  realv noise = atof(argv[9]);
  realv maxTrainedPC = atof(argv[10]);
  uint validationEveryNIter = atoi(argv[11]);
  cout << "Loading population " << endl;
  MixedEnsembles me;
  is >> me ;
  cout << "Loading datasets " << endl;
  ImageDataset trainingDataset;
  trainingDataset.load(trainingFile);
  SequenceClassDataset trainingClassDataset;
  trainingClassDataset.load(trainingClassFile);
  ImageDataset testDataset;
  testDataset.load(testFile);
  SequenceClassDataset testClassDataset;
  testClassDataset.load(testClassFile);
  cout << "Preparing training "<< endl;
  LearningParams params;
  params.setActualIteration(0);
  params.setMaxIterations(iterations);
  params.setLearningRate(learningRate);
  params.setMaxTrainedPercentage(maxTrainedPC);
  params.setSavedDuringProcess(true);
  params.setValidateEveryNIteration(validationEveryNIter);
  params.setNoise(noise);
  ofstream log("training.log");
  LayerCTC* ctc = (LayerCTC*)(me.getOutputNetwork()->getOutputLayer());
  ctc->setClassLabelIndex(trainingClassDataset.getClassLabelIndex());
  ctc->setClassLabels(trainingClassDataset.getClassLabels());
  MECTCTrainer trainer = MECTCTrainer(me,trainingDataset,trainingClassDataset,testDataset,testClassDataset,params,log);
  double t = (double) getTickCount();
  trainer.train();
  cout << "trained" << endl;
  t = ((double) getTickCount() - t) / getTickFrequency();
  cout << "Time :" << t << endl;
  ofs << me ;
  ofs.close();
  return EXIT_SUCCESS;
}
