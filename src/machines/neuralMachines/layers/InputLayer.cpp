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

void InputLayer::forward(FeatureVector signal){
  if(signal.getLength()!=numUnits){
    throw length_error("Wrong signal length");
  }
  inputSignal = signal;
  for (uint i = 0; i < numUnits; i++){
    if(stdev[i]==0){
      outputSignal[i]=(inputSignal[i]-mean[i]);
    }
    outputSignal[i]=(inputSignal[i]-mean[i])/stdev[i];
  }
  forward();
}

void InputLayer::forward(){
  list<Connection*>::iterator it;
  for(it=outputConnections.begin();it!=outputConnections.end();it++){
    (*it)->forward();
  }
}

void InputLayer::backward(ErrorVector deltas){

}

InputLayer::~InputLayer(){

}

ostream& operator<<(ostream& os, const InputLayer& l){
  os << "Input neuron layer : " << endl;
  os << "\t -Name :"<< l.getName() << endl;
  os << "\t -Units : "<< l.getNumUnits() << endl;
  os << "\t -Input Connections objects : " << l.getInputConnections().size() <<endl;
  os << "\t-Ouput Connections objects : "<< l.getOutputConnections().size() << endl;
  return os;
}
