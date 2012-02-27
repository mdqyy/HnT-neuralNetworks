/*!
 * \file PopulationBP.cpp
 * Body of the PopulationBP class.
 * \author Luc Mioulet
 */

#include "PopulationBP.hpp"

using namespace cv;
using namespace std;

PopulationBP::PopulationBP(PBDNN& _population, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask) : SupervisedTrainer(_population, _data, _featureMask, _indexMask), population(_population){

}

void PopulationBP::train(){
  uint i=0;
  do{
    i++;
    trainOneIteration();
    /*bpp.setLearningRate(bpp.getLearningRate()*bpp.getLearningRateDecrease());*/
  }while(i<100);
}

void PopulationBP::trainOneIteration(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  uint index=0;
  vector<FeatureVector> errors;
  vector<NeuralNetwork*> neuralNets=population.getPopulation();
  MSEMeasurer mse;
  realv iterationError=0;
  for(uint i=0; i<data.getNumSequences();i++){
    index=indexOrderSelection[i];
    for(uint j=0;j<data[index].size();j++){
      realv minError=10e+9;
      list<realv> scoreList;
      list<int> indexList;
      for(uint k=0;k<neuralNets.size();k++){
	neuralNets[k]->forward(data[index][j]);
	mse.totalError(neuralNets[k]->getOutputSignal(),data[index][j]);
	if(mse.getError()<minError){
	  minError=mse.getError();
	  scoreList.push_front(mse.getError());
	  indexList.push_front(k);
	}
	else{
	  scoreList.push_back(mse.getError());
	  indexList.push_back(k);
	}
      }
      list<int>::iterator iterIndex=indexList.begin();
      list<realv>::iterator iterScore=scoreList.begin();
      realv errorDifference=minError*0.05;
      while((*iterScore-minError)<errorDifference && iterScore!=scoreList.end()){
	neuralNets[*iterIndex]->backward(data[index][j], 0.001);
	iterIndex++;
	iterScore++;
      }
      iterationError+=minError;
    }
  }
  iterationError=iterationError/data.getNumSequences();
  cout << "Iteration error" <<iterationError << endl;
}

PopulationBP::~PopulationBP(){

}
