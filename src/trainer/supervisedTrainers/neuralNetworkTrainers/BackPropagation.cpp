/*!
 * \file BackPropagation.cpp
 * Body of the BackPropagation class.
 * \author Luc Mioulet
 */

#include "BackPropagation.hpp"
#include <iostream>

using namespace cv;
using namespace std;

BackPropagation::BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _data, LearningParams& _bpparams, Mask& _featureMask, Mask& _indexMask) : NeuralNetworkTrainer(_neuralNet, _data, _featureMask, _indexMask, _bpparams.isStochastic()), bpp(_bpparams){
  
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
  }while(i<bpp.getMaxIterations()/* && change>bpp.getMinChangeError()*/ && errorPerIteration[i-1]>bpp.getMinError());
}

void BackPropagation::trainOneIteration(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  cout << indexOrderSelection[0] << endl;
  uint index=0;
  realv error= 0;
  FeatureVector dataOutput;
  for(uint i=0; i<data.getNumSequences();i++){
    index=indexOrderSelection[i];
    if(neuralNet.isForward()){
      for(uint j=0;j<data[index].size() ; j++){
	neuralNet.forward(data[index][j]);
	FeatureVector target;
	if(bpp.getTask()!=BP_AUTOENCODER){
	  target=trainData.getTargetSample(index,j);
	}
	else{
	  target=data[index][j];
	}
	backward(target, bpp.getLearningRate()); 
	error+=measureSampleError(neuralNet.getOutputSignal(),target);
      }
    }
  }
  cout << error/data.getNumSequences() << endl;
  errorPerIteration.push_back(error/data.getNumSequences());
}

realv BackPropagation::measureSampleError(FeatureVector networkOutput, FeatureVector target){
  realv error = 0;
  if(bpp.getTask() == BP_CLASSIFICATION){
    ClassificationErrorMeasurer ce = ClassificationErrorMeasurer();
    error += ce.totalError(networkOutput,target);
  }
  else{  
    SEMeasurer mse;
    error += mse.totalError(networkOutput,target);
  }
  return error;
}

void BackPropagation::backward(FeatureVector _target, realv _learningRate){
  vector<ConnectionPtr> connections = neuralNet.getConnections();
  vector<LayerPtr> layers = neuralNet.getHiddenLayers();
  vector<ErrorVector> deltas = vector<ErrorVector>();/* pushed in inversed order so be careful */
  for(uint i = layers.size() -1 ; i > 0; i--){
    ValueVector derivatives = layers[i]->getDerivatives();
    if(i == layers.size()-1){
      deltas.push_back(calculateOutputDeltas(layers[i], _target, derivatives));
    }
    else{
      deltas.push_back(calculateDeltas(layers[i], _target, derivatives, deltas[deltas.size()-1]));
    }
  }
  for(uint i = 0; i < connections.size(); i++){
    updateConnection(connections[i], deltas[connections.size()-i-1],_learningRate);
  }
}

ErrorVector BackPropagation::calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta){
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for(uint i=0;i<delta.getLength();i++){
    delta[i]=_derivatives[i]*_layer->errorWeighting(_previousLayerDelta,_layer->getOutputConnection()->getWeightsFromNeuron(i));
  }
  return delta;
}

ErrorVector BackPropagation::calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives){
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for(uint i=0;i<_target.getLength();i++){
    delta[i] = _derivatives[i]*(_target[i]-_layer->getOutputSignal()[i]);  // error calculation if output layer
  }
  return delta;
}

void BackPropagation::updateConnection(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate){
  Mat weights = _connection->getWeights();
  for(int i=0;i<weights.rows;i++){
    for(int j=0;j<weights.cols;j++){
      weights.at<realv>(i,j)=weights.at<realv>(i,j)+_learningRate*_deltas[i]*_connection->getInputLayer()->getOutputSignal()[j];
    }
  }
  _connection->setWeights(weights);
}

BackPropagation::~BackPropagation(){

}
