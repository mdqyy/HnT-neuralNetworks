/*!
 * \file LayerSoftMax.cpp
 * Body of the LayerSoftMax class.
 * \author Luc Mioulet
 */

#include "LayerSoftMax.hpp"
#include <math.h>

using namespace std;
using namespace cv;

LayerSoftMax::LayerSoftMax() : Layer(){

}

LayerSoftMax::LayerSoftMax(uint _numUnits,string _name) : Layer(_numUnits,_name){

}

LayerSoftMax::LayerSoftMax(const LayerSoftMax& _clsm) : Layer(_clsm){

}

LayerSoftMax* LayerSoftMax::clone() const {
  return new LayerSoftMax(*this);
}

int LayerSoftMax::getLayerType() const{
  return LAYER_SOFTMAX;
}

void LayerSoftMax::forward(){
  outputSignal.reset(0.0);
  /* Accumulate neuron sum */
  FeatureVector layerInputSignal=getInputConnection()->getInputLayer()->getOutputSignal();
  forward(layerInputSignal);
  if(getOutputConnection()>0){
    getOutputConnection()->forward();
  }
}

void LayerSoftMax::forward(FeatureVector _signal){
  inputSignal = _signal;
  realv sum=0;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=signalWeighting(inputSignal, getInputConnection()->getWeightsToNeuron(i));
    sum+=signalWeighting(inputSignal, getInputConnection()->getWeightsToNeuron(i));
  }
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=outputSignal[i]/sum;
  }
  outputSignal[numUnits]=1.0;
}

void LayerSoftMax::backwardDeltas(bool _output, FeatureVector _target){
  deltas.reset(0.0);
  if(_output){
    for(uint i=0;i<deltas.getLength();i++){
      deltas[i] =_target[i]-outputSignal[i];  // error calculation if output layer
    }
  }
  else {
    throw logic_error("LayerSoftMax : This layer should only be an output");
  }
  getInputConnection()->getInputLayer()->backwardDeltas();
}

void LayerSoftMax::backwardWeights(realv _learningRate){
  getInputConnection()->backwardWeights(_learningRate);
}

LayerSoftMax::~LayerSoftMax(){

}

void LayerSoftMax::print(ostream& _os) const{
  _os << "SoftMax neuron layer : " << endl;
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& ofs, const LayerSoftMax& l){
  ofs << "< "<<l.getName()<<" "<<l.getNumUnits()<<" >"<<endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, LayerSoftMax& l){
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
