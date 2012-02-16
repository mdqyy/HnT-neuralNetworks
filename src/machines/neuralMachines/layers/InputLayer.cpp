/*!
 * \file InputLayer.cpp
 * Body of the InputLayer class.
 * \author Luc Mioulet
 */

#include "InputLayer.hpp"

using namespace cv;
using namespace std;

InputLayer::InputLayer(uint _numUnits, ValueVector _mean, ValueVector _stdev, string _name) : Layer(_numUnits, _name), mean(_mean), stdev(_stdev), inputSignal(_numUnits){

}

ValueVector InputLayer::getMean(){
  return mean;
}

ValueVector InputLayer::getStandardDeviation(){
  return stdev;
}

FeatureVector InputLayer::getInputSignal(){
  return inputSignal;
}

void InputLayer::setMean(ValueVector _mean){
  mean=_mean;
}

void InputLayer::setStandardDeviation(ValueVector _stdev){
  stdev=_stdev;
}

void InputLayer::forward(FeatureVector _signal){
  if(_signal.getLength()!=numUnits){
    throw length_error("Wrong signal length");
  }
  inputSignal = _signal;
  for (uint i = 0; i < numUnits; i++){
    if(stdev[i]==0.0){
      outputSignal[i]=(inputSignal[i]-mean[i]);
    }
    else{
      outputSignal[i]=(inputSignal[i]-mean[i])/stdev[i];
    }
  }
  outputSignal[numUnits]=1.0;
  forward();
}

void InputLayer::forward(){
  getOutputConnection()->forward();
}

void InputLayer::backwardDeltas(bool _output, FeatureVector _target){

}

void InputLayer::backwardWeights(realv _learningRate){

}

InputLayer::~InputLayer(){

}

ostream& operator<<(ostream& os, const InputLayer& l){
  os << "Input neuron layer : " << endl;
  os << "\t -Name :"<< l.getName() << endl;
  os << "\t -Units : "<< l.getNumUnits() << endl;
  return os;
}
