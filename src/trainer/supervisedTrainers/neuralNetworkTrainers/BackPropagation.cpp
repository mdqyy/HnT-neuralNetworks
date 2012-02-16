/*!
 * \file BackPropagation.cpp
 * Body of the BackPropagation class.
 * \author Luc Mioulet
 */

#include "BackPropagation.hpp"
#include <iostream>

using namespace cv;
using namespace std;

BackPropagation::BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _data, CrossValidationParams& _cvParams, BackPropParams& _bpparams) : NeuralNetworkTrainer(_neuralNet, _data, _cvParams, _bpparams.getDoStochastic()), bpp(_bpparams), errorPerIteration(vector<realv>()){ 
  
}

BackPropagation::BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _trainData, SupervisedDataset& _validationData, SupervisedDataset& _testData, CrossValidationParams& _cvParams,BackPropParams& _bpparams) : NeuralNetworkTrainer(_neuralNet, _trainData, _validationData, _testData, _cvParams,_bpparams.getDoStochastic()), bpp(_bpparams){

}

void BackPropagation::train(){
  uint i=0;
  realv change=bpp.getMinChangeError()+1.0;
  do{
    i++;
    trainOneIteration();
    if(i>2){
      change=abs(errorPerIteration[i-2]-errorPerIteration[i-1]);
    }
    bpp.setLearningRate(bpp.getLearningRate()*bpp.getLearningRateDecrease());
    cout << "Iteration : "<< i << " ; Error : " << errorPerIteration[i-1]<< endl;
  }while(i<bpp.getMaxIterations() && change>bpp.getMinChangeError() && errorPerIteration[i-1]>bpp.getMinError());
}

void BackPropagation::trainOneIteration(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  uint index=0;
  realv error= 0;
  realv ce=0;
  MSEMeasurer mse;
  ClassificationErrorMeasurer ceo(data.getNumSequences());
  FeatureVector dataOutput;
  for(uint i=0; i<data.getNumSequences();i++){
    index=indexOrderSelection[i];
    for(uint j=0;j<data[index].size() ; j++){
      neuralNet.forward(data[index][j]);
      error += mse.totalError(neuralNet.getOutputSignal(),trainData.getTargetSample(index,j));
      FeatureVector target=trainData.getTargetSample(index,j);
      neuralNet.backward(target, bpp.getLearningRate());
      ce +=ceo.totalError(neuralNet.getOutputSignal(),trainData.getTargetSample(index,j));
    }
  }
  cout << "classificatication error :" << ce <<endl;
  errorPerIteration.push_back(error/data.getNumSequences());
}

BackPropagation::~BackPropagation(){

}
