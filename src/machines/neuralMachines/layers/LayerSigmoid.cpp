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

LayerSigmoid::LayerSigmoid(uint _numUnits,string _name, bool _recurrent) : Layer(_numUnits,_name, _recurrent){

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

void LayerSigmoid::forward(FeatureVector _signal){
  networkInputSignal = _signal;
  FeatureVector inSig = createInputSignal();
  for(uint i=0;i<numUnits;i++){
    outputSignal[i]=1/(1+exp(-signalWeighting(inSig, getInputConnection()->getWeightsToNeuron(i))/*-recurrentSum*/ ));
  }
  outputSignal[numUnits]=1.0;
}

ValueVector LayerSigmoid::getDerivatives() const{
  ValueVector deriv = ValueVector(numUnits+1);
  for(uint i=0;i<deltas.getLength();i++){
    deriv[i] = outputSignal[i]*(1-outputSignal[i]); 
  }
  return deriv;
}

/*void LayerSigmoid::backwardDeltas(bool _output, FeatureVector _target){
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
  }*/

LayerSigmoid::~LayerSigmoid(){

}

void LayerSigmoid::print(ostream& _os) const{
  _os << "Sigmoid neuron layer : " << endl;
  if(isRecurrent()){
    _os << " \t -A recurrent layer." << endl;
  }
  _os << "\t -Name :"<<  getName() << endl;
  _os << "\t -Units : "<< getNumUnits() << endl;
}

ofstream& operator<<(ofstream& _ofs, const LayerSigmoid& _l){
  _ofs << "< "<< _l.getName()<<" "<< _l.getNumUnits()<<" "<< _l.isRecurrent();
  /*  if(_l.isRecurrent()){
    _ofs << " "<< *_l.getRecurrentConnection();
    }*/
  _ofs << " >"<<endl;
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, LayerSigmoid& _l){
  int nUnits, intRecurrent;
  bool boolRecurrent;
  ValueVector meanV, stdV;
  /*  Connection recCo;*/
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
  _l.setRecurent(boolRecurrent);
  return _ifs;
}
