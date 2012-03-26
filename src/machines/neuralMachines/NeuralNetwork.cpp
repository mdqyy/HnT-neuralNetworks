/*!
 * \file NeuralNetwork.cpp
 * Body of the NeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"
#include <sstream>


using namespace std;
using namespace cv;

NeuralNetwork::NeuralNetwork(InputLayer* _input,vector<Layer*> _hidden, Layer* _output, vector<Connection*> _connections, bool _forward,string _name) : NeuralMachine(_name), input(_input), hiddenLayers(_hidden), output(_output), connections(_connections), readForward(_forward){

}

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
    ostringstream temp;
    temp<<i;
    hiddenLayers[i]->setName(temp.str());
  }

  vector<Connection *> tempConnections = _cnn.getConnections();
  for (int i=0;i<tempConnections.size();i++){
    connections.push_back(tempConnections[i]->clone());
    connections[i]->setInputLayer(hiddenLayers[i]);
    hiddenLayers[i]->setOutputConnection(connections[i]);
    connections[i]->setOutputLayer(hiddenLayers[i+1]);
    hiddenLayers[i+1]->setInputConnection(connections[i]);
  }
}

NeuralNetwork* NeuralNetwork::clone() const{
  return new NeuralNetwork(*this);
}

InputLayer* NeuralNetwork::getInputLayer() const{
  return input;
}

vector<Layer*> NeuralNetwork::getHiddenLayers() const{
  return hiddenLayers;
}

vector<Connection*> NeuralNetwork::getConnections() const{
  return connections;
}

Layer* NeuralNetwork::getOutputLayer() const{
  return output;
}

FeatureVector NeuralNetwork::getInputSignal() const{
  return  input->getInputSignal();
}

FeatureVector NeuralNetwork::getOutputSignal() const{
  FeatureVector outputSig(output->getNumUnits());
  FeatureVector temp = output->getOutputSignal();
  for(uint i=0;i<outputSig.getLength();i++){
    outputSig[i]=temp[i];
  }
  return  outputSig;
}

bool NeuralNetwork::isForward() const{
  return readForward;
}

void NeuralNetwork::setInputLayer(InputLayer* _input){
  input=_input;
}

void NeuralNetwork::setHiddenLayers(std::vector<Layer*> _hidden){
  hiddenLayers=_hidden;
}

void NeuralNetwork::setOutputLayer(Layer* _output){
  output=_output;
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
  input->forward(_signal);
}

void NeuralNetwork::backward(FeatureVector _target,realv _learningRate){
  output->backwardDeltas(true, _target);
  output->backwardWeights( _learningRate);
}


NeuralNetwork::~NeuralNetwork(){

}
