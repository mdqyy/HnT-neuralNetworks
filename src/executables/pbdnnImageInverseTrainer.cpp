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
  arguments.push_back("population size");
  arguments.push_back("number of hidden units");
  arguments.push_back("learning dataset");
  arguments.push_back("validation dataset");
  arguments.push_back("ife scale");
  arguments.push_back("ife frame length");
  arguments.push_back("ife inter frame space");
  arguments.push_back("iterations");
  arguments.push_back("learning rate");
  arguments.push_back("input noise ratio");
  arguments.push_back("max trained percentage");
  arguments.push_back("validation step every N step");
  cout << helper("Pbdnn cluster", "Train a population of neural networks on a regression task using an image dataset.", arguments) << endl;
  if (argc != arguments.size() + 1) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
    return EXIT_FAILURE;
  }
  ImageDataset trainingDataset;
  ImageDataset testingDataset;
  trainingDataset.load(argv[3]);
  testingDataset.load(argv[4]);
  ImageFrameExtractor ife = ImageFrameExtractor(atof(argv[5]),atoi(argv[6]),atoi(argv[7]));
  cout << ife.getFrameSize() << " " << ife.getInterFrameSpace()<<" " <<ife.getScale() << endl;
  trainingDataset.setImageFrameExtractor(ife);
  testingDataset.setImageFrameExtractor(ife);
  cout << "Learning dataset loaded, total elements : " << trainingDataset.getNumberOfImages() << endl;
  cout << "Validation dataset loaded, total elements : " << testingDataset.getNumberOfImages() << endl;
  int populationSize = atoi(argv[1]);
  int numberOfHiddenUnits = atoi(argv[2]);
  AEMeasurer mae;
  int inputSize = trainingDataset.getFeatures(0)[0].getLength();
  PBDNN pop = PBDNN(populationSize, inputSize, numberOfHiddenUnits, ValueVector(inputSize), ValueVector(inputSize));
  //DiversityMeasurer diversity(pop, dataset2, mae,0.01);
  LearningParams params;
  params.setActualIteration(0);
  params.setMaxIterations(atoi(argv[8]));
  params.setLearningRate(atof(argv[9]));
  params.setMaxTrainedPercentage(atof(argv[11]));
  params.setDodges(1);
  params.setProximity(1.0);
  params.setSavedDuringProcess(true);
  params.setValidateEveryNIteration(atoi(argv[12]));
  params.setNoise(atof(argv[10]));
  ofstream log("training.log");
  ImagePopulationInverseTrainer ipbp(pop, trainingDataset, testingDataset,params,  log);
  // 07/02/13 : Not sure if useful or not so stop doing it
  /*cout << "Starting diversity" << endl << diversity.getDisagreementMatrix() << endl;
    cout << "Starting overall diversity : " << diversity.getDisagreementScalar() << endl;*/

  cout << "Training" << endl;
  double t = (double) getTickCount();
  ipbp.train();
  t = ((double) getTickCount() - t) / getTickFrequency();
  cout << "Time :" << t << endl;

  cout << endl << "Saving network" << endl;
  ofstream outStream("IAMpop.pop");
  outStream << pop;
  return EXIT_SUCCESS;
}
