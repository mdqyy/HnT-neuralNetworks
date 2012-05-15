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

LayerTanh::LayerTanh(uint _numUnits,string _name, bool _recurrent) : Layer(_numUnits,_name, _recurrent){

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

void LayerTanh::forward(FeatureVector _signal){
  inputNetworkSignal = _signal;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=tanh(signalWeighting(getInputSignal(), getInputConnection()->getWeightsToNeuron(i)));
  }
  outputSignal[numUnits]=1.0;
}

ValueVector LayerTanh::getDerivatives() const{
  ValueVector deriv = ValueVector(numUnits+1);
  for(uint i=0;i<deltas.getLength();i++){
    deriv[i] = (1-outputSignal[i]*outputSignal[i]); 
  }
  return deriv;
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


void LayerTanh::print(ostream& _os) const{
  _os << "Tanh neuron layer : " << endl;
  if(isRecurrent()){
    _os << " \t -A recurrent layer." << endl;
  }
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& _ofs, const LayerTanh& _l){
  _ofs << "< "<< _l.getName()<<" "<< _l.getNumUnits()<<" "<< _l.isRecurrent();
  /*  if(_l.isRecurrent()){
    _ofs << " "<< *_l.getRecurrentConnection();
    }*/
  _ofs << " >"<<endl;
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, LayerTanh& _l){
  int nUnits, intRecurrent;
  bool boolRecurrent;
  /* Connection recCo;*/
  ValueVector meanV, stdV;
  string name,temp;
  _ifs >> temp;
  _ifs >> name ;
  _ifs >> nUnits ;
  _ifs >> intRecurrent;
  boolRecurrent = intRecurrent == 1;
  /*  if(boolRecurrent) {
    _ifs >> recCo;
    _l.setRecurrentConnection(ConnectionPtr(new Connection(recCo)));
    }*/
  _ifs >> temp;
  _l.setName(name);
  _l.setNumUnits(nUnits);
  return _ifs;
}
