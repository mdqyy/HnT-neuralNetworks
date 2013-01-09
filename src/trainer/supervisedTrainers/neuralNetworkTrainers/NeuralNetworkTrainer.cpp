/*!
 * \file NeuralNetworkTrainer.cpp
 * Body of the NeuralNetworkTrainer class.
 * \author Luc Mioulet
 */

#include "NeuralNetworkTrainer.hpp"

using namespace std;
using namespace cv;

NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetwork& _neuralNet, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask, ostream& _log) : SupervisedTrainer(_neuralNet,_data,_featureMask,_indexMask, _log), neuralNet(_neuralNet){
  assert(neuralNet.getInputLayer()->getNumUnits()==data.getFeatureVectorLength());

}

NeuralNetworkTrainer::~NeuralNetworkTrainer(){

}
