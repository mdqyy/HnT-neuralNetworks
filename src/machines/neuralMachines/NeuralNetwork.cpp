/*!
 * \file NeuralNetwork.cpp
 * Body of the NeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"

using namespace std;
using namespace cv;

NeuralNetwork::NeuralNetwork(vector<LayerPtr> _hidden, vector<ConnectionPtr> _connections, bool _forward,string _name) : NeuralMachine(_name), hiddenLayers(_hidden), connections(_connections), readForward(_forward){
  
}
/*
NeuralNetwork::NeuralNetwork(const NeuralNetwork& _cnn) : NeuralMachine(_cnn.getName()+" copy"), input(_cnn.getInputLayer()->clone()),hiddenLayers(vector<Layer*>()), output(_cnn.getOutputLayer()->clone()), connections(vector<Connection*>()), readForward(_cnn.isForward()){
  vector<Layer *> tempLayers = _cnn.getHiddenLayers();
  for(int i=0;i<tempLayers.size();i++){
    if(i==0){
      hiddenLayers.push_back(input);
    }
    else if(i==tempLayers.size()-1){
      hiddenLayers.push_back(output);
    }
    else{
      hiddenLayers.push_back(tempLayers[i]->clone());
    }
  }
  vector<Connection *> tempConnections = _cnn.getConnections();
  for (int i=0;i<tempConnections.size();i++){
    connections.push_back(tempConnections[i]->clone());
    connections[i]->setInputLayer(hiddenLayers[i]);
    hiddenLayers[i]->setOutputConnection(connections[i]);
    connections[i]->setOutputLayer(hiddenLayers[i+1]);
    hiddenLayers[i+1]->setInputConnection(connections[i]);
  }
}*/

NeuralNetwork::NeuralNetwork(const NeuralNetwork& _cnn) : NeuralMachine(_cnn.getName()+" copy"),hiddenLayers(vector<LayerPtr>()), connections(vector<ConnectionPtr>()), readForward(_cnn.isForward()){
  vector<LayerPtr> tempLayers = _cnn.getHiddenLayers();
  for(int i=0;i<tempLayers.size();i++){
    hiddenLayers.push_back(LayerPtr(tempLayers[i]->clone()));
  }
  vector<ConnectionPtr> tempConnections = _cnn.getConnections();
  for (int i=0;i<tempConnections.size();i++){
    connections.push_back(ConnectionPtr(tempConnections[i]));
    connections[i]->setInputLayer(hiddenLayers[i].get());
    hiddenLayers[i]->setOutputConnection(connections[i].get());
    connections[i]->setOutputLayer(hiddenLayers[i+1].get());
    hiddenLayers[i+1]->setInputConnection(connections[i].get());
  }
}



NeuralNetwork* NeuralNetwork::clone() const{
  return new NeuralNetwork(*this);
}

Layer* NeuralNetwork::getInputLayer() const{
  return hiddenLayers[0].get();
}

vector<LayerPtr> NeuralNetwork::getHiddenLayers() const{
  return hiddenLayers;
}

vector<ConnectionPtr> NeuralNetwork::getConnections() const{
  return connections;
}

Layer* NeuralNetwork::getOutputLayer() const{
  return hiddenLayers[hiddenLayers.size()-1].get();
}

FeatureVector NeuralNetwork::getInputSignal() const{
  return  getInputLayer()->getInputSignal();
}

FeatureVector NeuralNetwork::getOutputSignal() const{
  FeatureVector outputSig(getOutputLayer()->getNumUnits());
  FeatureVector temp = getOutputLayer()->getOutputSignal();
  for(uint i=0;i<outputSig.getLength();i++){
    outputSig[i]=temp[i];
  }
  return  outputSig;
}

bool NeuralNetwork::isForward() const{
  return readForward;
}

void NeuralNetwork::setHiddenLayers(std::vector<LayerPtr> _hidden){
  hiddenLayers=_hidden;
}

void NeuralNetwork::forwardSequence(std::vector<FeatureVector> _sequence){
  if(readForward){
    for(uint i=0;i<_sequence.size();i++){
      forward(_sequence[i]);
    }
  }
  else{
    for(int i=_sequence.size()-1;i>=0;i--){
      forward(_sequence[i]);
    }
  }
}

void NeuralNetwork::forward(FeatureVector _signal){
  getInputLayer()->forward(_signal);
}

void NeuralNetwork::backward(FeatureVector _target,realv _learningRate){
  getOutputLayer()->backwardDeltas(true, _target);
  getOutputLayer()->backwardWeights( _learningRate);
}


NeuralNetwork::~NeuralNetwork(){

}
