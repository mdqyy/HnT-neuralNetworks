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

void LayerTanh::forward(){
  outputSignal.reset(0.0);
  /* Accumulate neuron sum */
  FeatureVector layerInputSignal=getInputConnection()->getInputLayer()->getOutputSignal();
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=tanh(signalWeighting(layerInputSignal, getInputConnection()->getWeightsToNeuron(i)));
  }
  outputSignal[numUnits]=1.0;
  if(getOutputConnection()>0){
    getOutputConnection()->forward();
  }
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


ostream& operator<<(ostream& os, const LayerTanh& l){
  os << "Tanh neuron layer : " << endl;
  os << "\t -Name :"<< l.getName() << endl;
  os << "\t -Units : "<<l.getNumUnits() << endl;
  return os;
}
