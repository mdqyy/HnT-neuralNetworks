#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>
#include <sstream>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main (int argc, char* argv[]){

	ClassificationDataset dataset;
	dataset.load(argv[4]);
	cout << "dataset loaded, total elements : "<< dataset.getNumSamples()<< endl ;
	int populationSize = atoi(argv[1]);
	int numberOfHiddenUnits =atoi(argv[2]);
	int iterations =atoi(argv[3]);
	string filename(argv[5]);
	PBDNN pop(populationSize,dataset.getFeatureVectorLength(),numberOfHiddenUnits,dataset.getMean(), dataset.getStandardDeviation());
	Mask mask;
	LearningParams params;
	params.setMaxIterations(iterations);
	params.setLearningRate(0.0001);
	params.setErrorToFirst(1.0);
	params.setErrorToFirstIncrease(1.0);
	params.setMaxTrained(populationSize);
	ofstream log("training.log");
	PopulationBP pbp(pop,dataset,params,mask,mask, log);
	double t = (double)getTickCount();
	pbp.train();
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Temps :" << t << endl;

	cout << endl <<"Saving network" << endl;
	ofstream outStream("IAMpop.txt");
	outStream << pop;

	vector<NeuralNetworkPtr> population= pop.getPopulation();
	ClassificationDataset dataset2;
	dataset2.load(argv[6]);
	cout << "Recording Data" << endl;
	for(uint i=0;i<population.size();i++){
		ostringstream name,path;
		name << "neuralNet" << i;
		UnsupervisedDataset nnData;
		nnData.setName(name.str());
		for(uint j=0;j<dataset2.getNumSequences();j++){
			vector<FeatureVector> features;
			for(uint k=0;k<dataset2[j].size() ; k++){
				population[i]->forward(dataset2[j][k]);
				features.push_back(population[i]->getOutputSignal());
			}
			nnData.addSequence(features);
		}
		path<<filename<<name.str()<<".xml";
		nnData.save(path.str());
	}

	return EXIT_SUCCESS;
}
