/*!
 * \file LayerTanh.cpp
 * Body of the LayerTanh class.
 * \author Luc Mioulet
 */

#include "LayerTanh.hpp"
#include <math.h>

using namespace std;
using namespace cv;

LayerTanh::LayerTanh() : Layer(){

}

LayerTanh::LayerTanh(uint _numUnits,string _name) : Layer(_numUnits,_name){

}

LayerTanh::LayerTanh(const LayerTanh& _clth) : Layer(_clth){

}

LayerTanh* LayerTanh::clone() const {
  return new LayerTanh(*this);
}

int LayerTanh::getLayerType() const{
  return LAYER_TANH;
}

void LayerTanh::forward(){
  outputSignal.reset(0.0);
  /* Accumulate neuron sum */
  FeatureVector layerInputSignal=getInputConnection()->getInputLayer()->getOutputSignal();
  forward(layerInputSignal);
  if(getOutputConnection()>0){
    getOutputConnection()->forward();
  }
}

void LayerTanh::forward(FeatureVector _signal){
  inputSignal = _signal;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=tanh(signalWeighting(inputSignal, getInputConnection()->getWeightsToNeuron(i)));
  }
  outputSignal[numUnits]=1.0;
}

void LayerTanh::backwardDeltas(bool _output, FeatureVector _target){
  deltas.reset(0.0);
  if(_output){
    for(uint i=0;i<deltas.getLength();i++){
      deltas[i] = (1-outputSignal[i]*outputSignal[i])*(_target[i]-outputSignal[i]);  // error calculation if output layer
    }
  }
  else {
    ErrorVector layerOutputError((getOutputConnection()->getOutputLayer())->getErrorVector().getLength()-1);
    for(uint j=0;j<layerOutputError.getLength() ;j++){
      layerOutputError[j]=getOutputConnection()->getOutputLayer()->getErrorVector()[j];
    }
    for(uint i=0;i<deltas.getLength();i++){
      deltas[i]=(1-outputSignal[i]*outputSignal[i])*errorWeighting(layerOutputError,getOutputConnection()->getWeightsFromNeuron(i)); // error calculation if non output layer
    }
  }
  getInputConnection()->getInputLayer()->backwardDeltas();
}

void LayerTanh::backwardWeights(realv _learningRate){
  getInputConnection()->backwardWeights(_learningRate);
}

LayerTanh::~LayerTanh(){

}


//ostream& operator<<(ostream& os, const LayerTanh& l){
void LayerTanh::print(ostream& _os) const{
  _os << "Tanh neuron layer : " << endl;
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& ofs, const LayerTanh& l){
  ofs << "< "<<l.getName()<<" "<<l.getNumUnits()<<" >"<<endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, LayerTanh& l){
  int nUnits;
  ValueVector meanV, stdV;
  string name,temp;
  ifs >> temp;
  ifs >> name ;
  ifs >> nUnits ;
  ifs >> temp;
  l.setName(name);
  l.setNumUnits(nUnits);
  return ifs;
}
