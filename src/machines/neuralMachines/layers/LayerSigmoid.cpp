/*!
 * \file LayerSigmoid.cpp
 * Body of the LayerSigmoid class.
 * \author Luc Mioulet
 */

#include "LayerSigmoid.hpp"
#include <math.h>

using namespace std;
using namespace cv;

LayerSigmoid::LayerSigmoid() : Layer() {

}

LayerSigmoid::LayerSigmoid(uint _numUnits,string _name) : Layer(_numUnits,_name){

}

LayerSigmoid::LayerSigmoid(const LayerSigmoid& _cls) : Layer(_cls){

}

LayerSigmoid* LayerSigmoid::clone() const{
  return new LayerSigmoid(*this);
}

int LayerSigmoid::getLayerType() const{
  return LAYER_SIGMOID;
}

void LayerSigmoid::forward(){
  outputSignal.reset(0.0);
  /* Accumulate neuron sum */
  FeatureVector layerInputSignal=getInputConnection()->getInputLayer()->getOutputSignal();
  forward(layerInputSignal);
  if(getOutputConnection()>0){
    getOutputConnection()->forward();
  }
}

void LayerSigmoid::forward(FeatureVector _signal){
  inputSignal = _signal;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=1/(1+exp(-signalWeighting(inputSignal, getInputConnection()->getWeightsToNeuron(i))));
  }
  outputSignal[numUnits]=1.0;
}

void LayerSigmoid::backwardDeltas(bool _output, FeatureVector _target){
  deltas.reset(0.0);
  if(_output){
    for(uint i=0;i<_target.getLength();i++){
      deltas[i] = outputSignal[i]*(1-outputSignal[i])*(_target[i]-outputSignal[i]);  // error calculation if output layer
    }
  }
  else {
    ErrorVector layerOutputError(getOutputConnection()->getOutputLayer()->getErrorVector().getLength()-1);
    for(uint j=0;j<layerOutputError.getLength() ;j++){
      layerOutputError[j]=getOutputConnection()->getOutputLayer()->getErrorVector()[j];
    }
    for(uint i=0;i<deltas.getLength();i++){
      deltas[i]=outputSignal[i]*(1-outputSignal[i])*errorWeighting(layerOutputError,getOutputConnection()->getWeightsFromNeuron(i)); // error calculation if non output layer
    }
  }
  getInputConnection()->getInputLayer()->backwardDeltas();
}

void LayerSigmoid::backwardWeights(realv _learningRate){
  getInputConnection()->backwardWeights(_learningRate);
}

LayerSigmoid::~LayerSigmoid(){

}

void LayerSigmoid::print(ostream& _os) const{
  _os << "Sigmoid neuron layer : " << endl;
  _os << "\t -Name :"<<  getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}
ofstream& operator<<(ofstream& ofs, const LayerSigmoid& l){
  ofs << "< "<<l.getName()<<" "<<l.getNumUnits()<<" >"<<endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, LayerSigmoid& l){
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
