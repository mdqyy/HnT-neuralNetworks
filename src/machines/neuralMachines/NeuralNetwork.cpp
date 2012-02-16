/*!
 * \file NeuralNetwork.cpp
 * Body of the NeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork(InputLayer& _input,list<Layer*> _hidden, Layer& _output, list<Connection*> _connections, string _name) : NeuralMachine(_name), input(_input), hiddenLayers(_hidden), output(_output), connections(_connections){

}

InputLayer& NeuralNetwork::getInputLayer(){
  return input;
}

list<Layer*> NeuralNetwork::getHiddenLayers(){
  return hiddenLayers;
}

Layer& NeuralNetwork::getOutputLayer(){
  return output;
}

FeatureVector NeuralNetwork::getInputSignal(){
  return  input.getInputSignal();
}

FeatureVector NeuralNetwork::getOutputSignal(){
  FeatureVector outputSig(output.getNumUnits());
  FeatureVector temp = output.getOutputSignal();
  for(uint i=0;i<outputSig.getLength();i++){
    outputSig[i]=temp[i];
  }
  return  outputSig;
}

void NeuralNetwork::setInputLayer(InputLayer& _input){
  input=_input;
}

void NeuralNetwork::setHiddenLayers(std::list<Layer*> _hidden){
  hiddenLayers=_hidden;
}

void NeuralNetwork::setOutputLayer(Layer& _output){
  output=_output;
}

void NeuralNetwork::forward(FeatureVector _signal){
  input.forward(_signal);
}

void NeuralNetwork::backward(FeatureVector _target,realv _learningRate){
  output.backwardDeltas(true, _target);
  output.backwardWeights( _learningRate);
}


NeuralNetwork::~NeuralNetwork(){

}
