/*!
 * \file Layer.cpp
 * Body of the Layer class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"

using namespace std;
using namespace cv;

Layer::Layer() : Machine("Layer"), numUnits(1),inputConnection(0), outputConnection(0), outputSignal(FeatureVector(2)), deltas(ErrorVector(2)), networkInputSignal(1), recurrent(false), inputSignals(std::vector<FeatureVector>()), outputSignals(std::vector<FeatureVector>()), inputSignal(0){
  
}

Layer::Layer(uint _numUnits, string _name,bool _recurrent) : Machine(_name), numUnits(_numUnits),inputConnection(0), outputConnection(0), outputSignal(FeatureVector(_numUnits+1)), deltas(ErrorVector(_numUnits+1)), networkInputSignal(_numUnits), recurrent(_recurrent),inputSignals(std::vector<FeatureVector>()),outputSignals(std::vector<FeatureVector>()), inputSignal(0){
  assert(numUnits>0);
}
  
Layer::Layer(const Layer& _cl) : Machine(_cl.getName()), numUnits(_cl.getNumUnits()),   inputConnection(0), outputConnection(0), outputSignal(FeatureVector(numUnits+1)), deltas(ErrorVector(numUnits+1)), recurrent(_cl.isRecurrent()),inputSignals(std::vector<FeatureVector>()),outputSignals(std::vector<FeatureVector>()), inputSignal(0){
  if(_cl.getInputConnection() !=0){
    inputConnection = _cl.getInputConnection();
  }
  if(_cl.getOutputConnection()!=0){
    outputConnection = _cl.getOutputConnection();
  }
}

uint Layer::getNumUnits() const{
  return numUnits;
}

Connection* Layer::getInputConnection() const{
  return inputConnection;
}

Connection* Layer::getOutputConnection() const{
  return outputConnection;
}

vector<FeatureVector> Layer::getOutputSignals() const{
  return outputSignals;
}

vector<FeatureVector> Layer::getInputSignals() const{
  return inputSignals;
}

bool Layer::isRecurrent() const{
  return recurrent;
}

FeatureVector Layer::getNetworkInputSignal() const{
  return networkInputSignal;
}

FeatureVector Layer::createInputSignal() const{
  FeatureVector inSig;
  if(isRecurrent()){
    inSig = FeatureVector(inputConnection->getWeights().cols);
    for(uint i=0;i<networkInputSignal.getLength();i++){
      inSig[i] = networkInputSignal[i];
    }
    for(uint i=0;i<outputSignal.getLength()-1;i++){ /* -1 since we do not use the bias of the output only the input*/
      inSig[networkInputSignal.getLength()+i] = getOutputSignal()[i];
    }
  }
  else{
    inSig = networkInputSignal;
  }
  this->inputSignal = inSig;
  return inSig;
}

FeatureVector Layer::getInputSignal() const{
	return inputSignal;
}

FeatureVector Layer::getOutputSignal() const{
  return outputSignal;
}

ErrorVector Layer::getErrorVector() const{
  return deltas;
}

void Layer::setNumUnits(uint _numUnits){
  numUnits=_numUnits;
}

void Layer::setErrorVector(ErrorVector _deltas){
  deltas=_deltas;
}

void Layer::setInputConnection(Connection* _connection){
  inputConnection=_connection;
}

void Layer::forwardSequence(std::vector<FeatureVector> _sequence){
  for(uint i=0; i < _sequence.size();i++){
	  forward(_sequence[i]);
	  inputSignals.push_back(this->getInputSignal());
	  outputSignals.push_back(this->getOutputSignal());
  }
}

void Layer::setOutputConnection(Connection* _connection){
  outputConnection=_connection;
}

void Layer::setRecurent(bool _state){
  recurrent = _state;
}

realv Layer::signalWeighting(FeatureVector _signal, Mat _weights){
  if(_signal.getLength()!=((uint)_weights.cols)){
    throw length_error("Layer : Uncorrect length between signal and weights");
  }
  realv sum=0;
  for(uint i=0;i<_signal.getLength();i++){
    sum+=_signal[i]*_weights.at<realv>(0,i);
  }
  return sum;
}

realv Layer::errorWeighting(ErrorVector _deltas, Mat _weights){
  if(_deltas.getLength()!=((uint)_weights.rows)){
    throw length_error("Layer : Uncorrect length between deltas and weights");
  }
  realv sum=0;
  for(uint i=0;i<_deltas.getLength();i++){
    sum+=_deltas[i]*_weights.at<realv>(i,0);
  }
  return sum;
}

void Layer::reset(){
  outputSignals.clear();
  inputSignals.clear();
}

Layer::~Layer(){
  inputConnection=0;
  outputConnection = 0;
}


