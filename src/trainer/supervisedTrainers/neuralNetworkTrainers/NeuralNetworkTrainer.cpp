/*!
 * \file NeuralNetworkTrainer.cpp
 * Body of the NeuralNetworkTrainer class.
 * \author Luc Mioulet
 */

#include "NeuralNetworkTrainer.hpp"

using namespace std;
using namespace cv;

NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetwork& _neuralNet, SupervisedDataset& _data, CrossValidationParams _cvParams, bool _doStochastic) : SupervisedTrainer(_neuralNet,_data,_cvParams), neuralNet(_neuralNet), doStochastic(_doStochastic){

}

NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetwork& _machine, SupervisedDataset& _trainData, SupervisedDataset& _validationData, SupervisedDataset& _testData, CrossValidationParams& _cvParams, bool _doStochastic) : SupervisedTrainer(_machine,_trainData,_validationData, _testData, _cvParams), neuralNet(_machine), doStochastic(_doStochastic){

}

vector<uint> NeuralNetworkTrainer::defineIndexOrderSelection(uint _numSequences){
  vector<uint> indexOrder;
  for(uint i=0 ;  i<_numSequences; i++){
    indexOrder.push_back(i);
  }
  if(doStochastic){
    int exchangeIndex=0;
    for(uint i=0 ;  i<_numSequences; i++){
      RNG random;
      exchangeIndex=random.uniform(0,_numSequences);
      indexOrder.push_back(i);
      swap(indexOrder[i],indexOrder[exchangeIndex]);
    } 
  }
  return indexOrder;
}

NeuralNetworkTrainer::~NeuralNetworkTrainer(){

}
