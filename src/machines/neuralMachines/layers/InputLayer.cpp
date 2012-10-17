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

InputLayer::InputLayer(uint _numUnits, const ValueVector _mean, const ValueVector _stdev, string _name) : Layer(_numUnits, _name), meanVector(_mean), stdevVector(_stdev){

}

InputLayer::InputLayer(const InputLayer& _cil) : Layer(_cil){
  meanVector = _cil.getMean();
  stdevVector = _cil.getStandardDeviation();
}

InputLayer* InputLayer::clone() const{
  return new InputLayer(*this);
}

int InputLayer::getLayerType() const{
  return LAYER_INPUT;
}

ValueVector InputLayer::getMean() const{
  return meanVector;
}

ValueVector InputLayer::getStandardDeviation() const{
  return stdevVector;
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
  networkInputSignal = _signal;
  for (uint i = 0; i < numUnits; i++){
    if(stdevVector[i]==0.0){
      outputSignal[i]=(networkInputSignal[i]-meanVector[i]);
    }
    else{
      outputSignal[i]=(networkInputSignal[i]-meanVector[i])/stdevVector[i];
    }
  }
  outputSignal[numUnits]=1.0;
  forward();
}

void InputLayer::forward(){
  getOutputConnection()->forward();
}

ValueVector InputLayer::getDerivatives() const{
  ValueVector deriv = ValueVector(numUnits+1);
  return deriv;
}

/*void InputLayer::backwardDeltas(bool _output, FeatureVector _target){

}

void InputLayer::backwardWeights(realv _learningRate){

}*/

InputLayer::~InputLayer(){

}

//ostream& operator<<(ostream& os, const InputLayer& l){
void InputLayer::print(std::ostream& _os) const{
  _os << "Input neuron layer : " << endl;
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& ofs, const InputLayer& l){
  ofs << " < "<<l.getName()<<" "<<l.getNumUnits()<<" ";
  ofs << l.getMean()<<" ";
  ofs << l.getStandardDeviation()<<" > "<<endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, InputLayer& l){
  int nUnits;
  ValueVector meanV, stdV;
  string name,temp;
  ifs >> temp;
  ifs >> name ;
  ifs >> nUnits ;
  ifs >> meanV ;
  ifs >> stdV;
  ifs >> temp;
  l.setName(name);
  l.setNumUnits(nUnits);
  l.setMean(meanV);
  l.setStandardDeviation(stdV);
  return ifs;
}
