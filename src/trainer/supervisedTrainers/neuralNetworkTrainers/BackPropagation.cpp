/*!
 * \file BackPropagation.cpp
 * Body of the BackPropagation class.
 * \author Luc Mioulet
 */

#include "BackPropagation.hpp"
#include "../../errorMeasurers/MSEMeasurer.hpp"

using namespace cv;
using namespace std;

BackPropagation::BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _data, CrossValidationParams _cvParams, bool _doStochastic) : NeuralNetworkTrainer(_neuralNet, _data, _cvParams, _doStochastic){ 
  
}

BackPropagation(NeuralNetwork& _machine, SupervisedDataset& _trainData, SupervisedDataset& _validationData, SupervisedDataset& _testData, CrossValidationParams& _cvParams, bool _doStochastic) : NeuralNetworkTrainer(_neuralNet, _trainData, _validationData, _testData, _cvParams, _doStochastic){

}

void BackPropagation::train(){
  
}

void BackPropagation::trainOneIteration(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  uint index=0;
  MSEMeasurer mse;
  FeatureVector dataOutput;
  for(uint i=0; i<data.getNumSequences();i++){
    index=indexOrderSelection[i];
    if(data[index].size()!=data.getSequenceClassesIndex(index)){
      throw length_error(" Class length and sequence length are different");
    }
    for(uint j=0;j<data[index].size() ; j++){
      neuralNet.forward(data[index][j]);
      dataOutput = data;
      neuralNet.backward(mse.measureErrorPerUnit(machine.getOutputSignal()));
    }
  }
}

BackPropagation::~BackPropagation(){

}
