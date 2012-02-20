/*!
 * \file Layer.cpp
 * Body of the Layer class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"

using namespace std;
using namespace cv;

Layer::Layer(uint _numUnits, string _name) : Machine(_name), numUnits(_numUnits),inputConnection(0), outputConnection(0), outputSignal(FeatureVector(_numUnits+1)), deltas(ErrorVector(_numUnits+1)){
  assert(numUnits>0);

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

FeatureVector Layer::getOutputSignal() const{
  return outputSignal;
}

ErrorVector Layer::getErrorVector() const{
  return deltas;
}

void Layer::setErrorVector(ErrorVector _deltas){
  deltas=_deltas;
}

void Layer::setInputConnection(Connection* _connection){
  inputConnection=_connection;
}

void Layer::forwardSequence(std::vector<FeatureVector> _sequence){

}

void Layer::setOutputConnection(Connection* _connection){
  outputConnection=_connection;
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

Layer::~Layer(){

}
