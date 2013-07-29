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

LayerSoftMax::LayerSoftMax(uint _numUnits,string _name, bool _recurrent) : Layer(_numUnits,_name, _recurrent){

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
  if(!isRecurrent()){
    outputSignal.reset(0.0);
  }
  /* Accumulate neuron sum */
  FeatureVector layerInputSignal=getInputConnection()->getInputLayer()->getOutputSignal();
  forward(layerInputSignal);
  if(getOutputConnection()>0){
    getOutputConnection()->forward();
  }
}

void LayerSoftMax::forward(FeatureVector _signal){
  networkInputSignal = _signal;
  realv sum=0;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=exp(-signalWeighting(createInputSignal(), getInputConnection()->getWeightsToNeuron(i)));
    sum += outputSignal[i];/*signalWeighting(createInputSignal(), getInputConnection()->getWeightsToNeuron(i));*/
  }
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=outputSignal[i]/sum;
  }
  outputSignal[numUnits]=1.0;
}

ValueVector LayerSoftMax::getDerivatives() const{
  ValueVector deriv = ValueVector(numUnits+1);
  for(uint i = 0 ; i < deriv.getLength() ; i++){
    deriv[i] = outputSignal[i]*(1-outputSignal[i]);
  }
  return deriv;
}

LayerSoftMax::~LayerSoftMax(){

}

void LayerSoftMax::print(ostream& _os) const{
  _os << "SoftMax neuron layer : " << endl;
  if(isRecurrent()){
    _os << "\t -A recurrent layer." << endl;
  }
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& _ofs, const LayerSoftMax& _l){
  _ofs << " < "<< _l.getName()<<" "<< _l.getNumUnits()<<" "<< _l.isRecurrent();
  _ofs << " > "<<endl;
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, LayerSoftMax& _l){
  int nUnits;
  bool boolRecurrent;
  ValueVector meanV, stdV;
  /*  Connection recCo; */
  string name,temp;
  _ifs >> temp;
  _ifs >> name ;
  _ifs >> nUnits ;
  _ifs >> boolRecurrent;
  _ifs >> temp;
  _l.setName(name);
  _l.setNumUnits(nUnits);
  _l.setRecurent(boolRecurrent);
  return _ifs;
}
