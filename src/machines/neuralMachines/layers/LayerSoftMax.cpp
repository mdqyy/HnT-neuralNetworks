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
  inputNetworkSignal = _signal;
  realv sum=0;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=signalWeighting(getInputSignal(), getInputConnection()->getWeightsToNeuron(i));
    sum += signalWeighting(getInputSignal(), getInputConnection()->getWeightsToNeuron(i));
  }
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=outputSignal[i]/sum;
  }
  outputSignal[numUnits]=1.0;
}

ValueVector LayerSoftMax::getDerivatives() const{
  ValueVector deriv = ValueVector(numUnits+1);
  for(uint i = 0 ; i < deriv.getLength() ; i++){
    deriv[i] = 1.0;
  }
  return deriv;
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
  if(isRecurrent()){
    _os << "\t -A recurrent layer." << endl;
  }
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& _ofs, const LayerSoftMax& _l){
  _ofs << "< "<< _l.getName()<<" "<< _l.getNumUnits()<<" "<< _l.isRecurrent();
  /*  if(_l.isRecurrent()){
    _ofs << " "<< *_l.getRecurrentConnection();
    }*/
  _ofs << " >"<<endl;
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, LayerSoftMax& _l){
  int nUnits, intRecurrent;
  bool boolRecurrent;
  ValueVector meanV, stdV;
  /*  Connection recCo; */
  string name,temp;
  _ifs >> temp;
  _ifs >> name ;
  _ifs >> nUnits ;
  _ifs >> intRecurrent;
  boolRecurrent = intRecurrent == 1;
  /*if(boolRecurrent) {
    _ifs >> recCo;
    _l.setRecurrentConnection(ConnectionPtr(new Connection(recCo)));
    }*/
  _ifs >> temp;
  _l.setName(name);
  _l.setNumUnits(nUnits);
  return _ifs;
}
