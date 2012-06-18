/*!
 * \file LayerCTC.cpp
 * Body of the LayerCTC class.
 * \author Luc Mioulet
 */

#include "LayerCTC.hpp"
#include <math.h>

using namespace std;
using namespace cv;

LayerCTC::LayerCTC() : Layer(){

}

LayerCTC::LayerCTC(uint _numUnits,string _name) : Layer(_numUnits,_name, false){

}

LayerCTC::LayerCTC(const LayerCTC& _clsm) : Layer(_clsm){

}

LayerCTC* LayerCTC::clone() const {
  return new LayerCTC(*this);
}

int LayerCTC::getLayerType() const{
  return LAYER_CTC;
}

void LayerCTC::forward(){
  outputSignal.reset(0.0);
  /* Accumulate neuron sum */
  FeatureVector layerInputSignal=getInputConnection()->getInputLayer()->getOutputSignal();
  forward(layerInputSignal);
  if(getOutputConnection()>0){
    getOutputConnection()->forward();
  }
}

void LayerCTC::forward(FeatureVector _signal){
  networkInputSignal = _signal;
  realv sum=0;
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=signalWeighting(createInputSignal(), getInputConnection()->getWeightsToNeuron(i));
    sum += signalWeighting(createInputSignal(), getInputConnection()->getWeightsToNeuron(i));
  }
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=outputSignal[i]/sum;
  }
  outputSignal[numUnits]=1.0;
}

ValueVector LayerCTC::getDerivatives() const{
  ValueVector deriv = ValueVector(numUnits+1);
  for(uint i = 0 ; i < deriv.getLength() ; i++){
    deriv[i] = 1.0;
  }
  return deriv;
}

vector<ValueVector> LayerCTC::getDerivatives(vector<FeatureVector> _forwardVariables, vector<FeatureVector> _backwardVariables) const{
	vector<ValueVector> derivatives = vector<ValueVector>();
	return derivatives;
}

LayerCTC::~LayerCTC(){

}

void LayerCTC::print(ostream& _os) const{
  _os << "CTC neuron layer : " << endl;
  if(isRecurrent()){
    _os << "\t -A recurrent layer." << endl;
  }
  _os << "\t -Name :"<< getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& _ofs, const LayerCTC& _l){
  _ofs << "< "<< _l.getName()<<" "<< _l.getNumUnits()<<" "<< _l.isRecurrent();
  /*  if(_l.isRecurrent()){
    _ofs << " "<< *_l.getRecurrentConnection();
    }*/
  _ofs << " >"<<endl;
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, LayerCTC& _l){
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
