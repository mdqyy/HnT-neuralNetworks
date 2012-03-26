/*!
 * \file InputLayer.cpp
 * Body of the InputLayer class.
 * \author Luc Mioulet
 */

#include "InputLayer.hpp"

using namespace cv;
using namespace std;

InputLayer::InputLayer() : Layer(), meanVector(FeatureVector(0)), stdevVector(FeatureVector(0)){

}

InputLayer::InputLayer(uint _numUnits, const ValueVector _mean, const ValueVector _stdev, string _name) : Layer(_numUnits, _name), meanVector(_mean), stdevVector(_stdev), inputSignal(_numUnits){

}

InputLayer::InputLayer(const InputLayer& _cil) : Layer(_cil){
  meanVector = _cil.getMean();
  stdevVector = _cil.getStandardDeviation();
}

InputLayer* InputLayer::clone() const{
  return new InputLayer(*this);
}

ValueVector InputLayer::getMean() const{
  return meanVector;
}

ValueVector InputLayer::getStandardDeviation() const{
  return stdevVector;
}

FeatureVector InputLayer::getInputSignal(){
  return inputSignal;
}

void InputLayer::setMean(ValueVector _mean){
  meanVector=_mean;
}

void InputLayer::setStandardDeviation(ValueVector _stdev){
  stdevVector=_stdev;
}

void InputLayer::forward(FeatureVector _signal){
  if(_signal.getLength()!=numUnits){
    throw length_error("InputLayer : Wrong signal length");
  }
  inputSignal = _signal;
  for (uint i = 0; i < numUnits; i++){
    if(stdevVector[i]==0.0){
      outputSignal[i]=(inputSignal[i]-meanVector[i]);
    }
    else{
      outputSignal[i]=(inputSignal[i]-meanVector[i])/stdevVector[i];
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
