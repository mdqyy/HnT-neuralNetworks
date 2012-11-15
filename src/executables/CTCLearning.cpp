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
	vector<string> arguments;
	arguments.push_back("dataset used for training");
	arguments.push_back("number of classes (with no label)");
	arguments.push_back("training iterations");
	helper("CTC Learning", "Create a CTC architecture and train it on a dataset.",arguments);
	ClassificationDataset dataset;
	dataset.load(argv[1]);
	cout << "dataset loaded, total elements : "<< dataset.getNumSamples()<< endl ;
	int numClasses = dataset.getNumberOfClasses();
	int numInputs = dataset[0][0].getLength();
	ValueVector mean = dataset.getMean();
	ValueVector stdDev = dataset.getStandardDeviation();
	LayerCTC ctc = LayerCTC(numClasses,"ctc");
	InputLayer input = InputLayer(numInputs,mean,stdDev,"input");

	Mat weights = Mat(numInputs,numClasses,CV_64FC1);
	Connection ctcConnection = Connection(0,&ctc,weights);
	ctcConnection.initializeWeights(1,0,1.0);
	Mask mask;
	CTCTrainer ctcTrainer = CTCTrainer(ctc, dataset,mask, mask);
	ctcTrainer.train();
	/*
	params.setMaxIterations(iterations);
	params.setLearningRate(0.001);

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
	}*/

	return EXIT_SUCCESS;
}
