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
  return  output.getOutputSignal();
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

void NeuralNetwork::forward(FeatureVector signal){
  input.forward(signal);
}

void NeuralNetwork::backward(ErrorVector deltas){

}


NeuralNetwork::~NeuralNetwork(){

}
