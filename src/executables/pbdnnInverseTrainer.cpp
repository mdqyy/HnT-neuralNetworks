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
  arguments.push_back("number of iterations");
  arguments.push_back("learning dataset");
  arguments.push_back("validation dataset");
  arguments.push_back("simple load mode");
  cout << helper("Pbdnn cluster", "Train a population of neural networks on a regression task.", arguments) << endl;
  if (argc != arguments.size() + 1) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() << " required" << endl;
    return EXIT_FAILURE;
  }
  int simpleMode = atoi(argv[6]);
  RegressionDataset dataset;
  RegressionDataset dataset2;
  if(simpleMode!=0) {
    dataset.simpleLoad(argv[4]);
    dataset2.simpleLoad(argv[5]);
  }
  else {
    dataset.load(argv[4]);
    dataset2.load(argv[5]);
  }
  cout << "Learning dataset loaded, total elements : " << dataset.getNumSamples() << endl;
  cout << "Validation dataset loaded, total elements : " << dataset2.getNumSamples() << endl;
  int populationSize = atoi(argv[1]);
  int numberOfHiddenUnits = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  AEMeasurer mae;
  PBDNN pop = PBDNN(populationSize, dataset.getFeatureVectorLength(), numberOfHiddenUnits, dataset.getMean(), dataset.getStandardDeviation());
  DiversityMeasurer diversity(pop, dataset2, mae,0.01);

  // 07/02/13 : Not sure if useful or not so stop doing it
  /*do {
    pop = PBDNN(populationSize, dataset.getFeatureVectorLength(), numberOfHiddenUnits, dataset.getMean(), dataset.getStandardDeviation());
    diversity.measurePerformance();
    } while (diversity.getDisagreementScalar() < 0.17);*/
  Mask mask;
  LearningParams params;
  params.setActualIteration(0);
  params.setMaxIterations(iterations);
  params.setLearningRate(0.001);
  params.setMaxTrainedPercentage(0.1);
  params.setDodges(1);
  params.setProximity(1.0);
  params.setSavedDuringProcess(true);
  params.setValidateEveryNIteration(100);
  ofstream log("training.log");
  PopulationInverseTrainer pbp(pop, dataset, params, dataset2, mask, mask, log);
  // 07/02/13 : Not sure if useful or not so stop doing it
  /*cout << "Starting diversity" << endl << diversity.getDisagreementMatrix() << endl;
    cout << "Starting overall diversity : " << diversity.getDisagreementScalar() << endl;*/

  cout << "Training" << endl;
  double t = (double) getTickCount();
  pbp.train();
  t = ((double) getTickCount() - t) / getTickFrequency();
  cout << "Time :" << t << endl;

  cout << endl << "Saving network" << endl;
  ofstream outStream("IAMpop.pop");
  outStream << pop;

  if(simpleMode == 0) {
    cout << "Recording Data" << endl;
    vector<NeuralNetworkPtr> population = pop.getPopulation();
    vector<Vec3b> colors = createColorRepartition(population.size());
    vector<vector<int> > assignedTo = diversity.findBestNetwork();
    vector<vector<FeatureVector> > recomposed = diversity.buildBestOutput();
    vector<int> pngParams = vector<int>();
    pngParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
    pngParams.push_back(3);

    for (uint i = 0; i < population.size(); i++) {
      ostringstream dir;
      dir << "network" << i;
      if (mkdir(dir.str().c_str(), S_IRWXU) == 0) {
	for (uint j = 0; j < dataset2.getNumSequences(); j++) {
	  ostringstream name;
	  name << "network" << i << "\/neuralNet" << i << "sample" << j << ".png";
	  vector<FeatureVector> features;
	  for (uint k = 0; k < dataset2[j].size(); k++) {
	    population[i]->forward(dataset2[j][k]);
	    features.push_back(population[i]->getOutputSignal());
	  }
	  vector<int> color = vector<int>(features.size(), i);
	  Mat image = buildColorMapImage(features, 3, color, colors);
	  imwrite(name.str(), image, pngParams);
	}
      }
      else {
	throw invalid_argument("pbdnnCluster : could not create directory");
      }
    }

    ostringstream dirR;
    dirR << "recomposed";
    if (mkdir(dirR.str().c_str(), S_IRWXU) == 0) {
      for (uint j = 0; j < recomposed.size(); j++) {
	ostringstream name;
	name << "recomposed\/recomposedSample" << j << ".png";
	vector<FeatureVector> features;
	Mat image = buildColorMapImage(recomposed[j], 3, assignedTo[j], colors);
	imwrite(name.str(), image, pngParams);
      }
    }
  }
  return EXIT_SUCCESS;
}
