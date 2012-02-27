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

NeuralNetworkTrainer::~NeuralNetworkTrainer(){

}
