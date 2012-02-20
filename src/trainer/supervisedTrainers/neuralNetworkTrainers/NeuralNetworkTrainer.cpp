/*!
 * \file NeuralNetworkTrainer.cpp
 * Body of the NeuralNetworkTrainer class.
 * \author Luc Mioulet
 */

#include "NeuralNetworkTrainer.hpp"

using namespace std;
using namespace cv;

NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetwork& _neuralNet, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask, bool _doStochastic) : SupervisedTrainer(_neuralNet,_data,_featureMask,_indexMask), neuralNet(_neuralNet), doStochastic(_doStochastic){
  assert(neuralNet.getInputLayer().getNumUnits()==data.getFeatureVectorLength());

}

vector<uint> NeuralNetworkTrainer::defineIndexOrderSelection(uint _numSequences){
  vector<uint> indexOrder;
  for(uint i=0 ;  i<_numSequences; i++){
    indexOrder.push_back(i);
  }
  if(doStochastic){
    int exchangeIndex=0;
    RNG random;
    random.next();
    for(uint i=0 ;  i<_numSequences; i++){	 
      exchangeIndex=random.uniform(0,_numSequences);
      swap(indexOrder[i],indexOrder[exchangeIndex]);
    } 
  }
  return indexOrder;
}

NeuralNetworkTrainer::~NeuralNetworkTrainer(){

}
