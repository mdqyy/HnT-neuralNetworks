/*!
 * \file Layer.cpp
 * Body of the Layer class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"

using namespace std;
using namespace cv;

Layer::Layer(uint _numUnits, string _name) : Machine(_name), numUnits(_numUnits), inputConnections(std::list<Connection*>()), outputConnections(std::list<Connection*>()), outputSignal(FeatureVector(_numUnits)), deltas(ErrorVector(_numUnits)){
  assert(numUnits>0);

}

uint Layer::getNumUnits() const{
  return numUnits;
}

list<Connection*> Layer::getInputConnections() const{
  return inputConnections;
}

list<Connection*> Layer::getOutputConnections() const{
  return outputConnections;
}

FeatureVector Layer::getOutputSignal() const{
  return outputSignal;
}

ErrorVector Layer::getErrorVector() const{
  return deltas;
}

void Layer::setInputConnections(list<Connection*> connections){
  /*  if(connections.length()!=numUnits){
    throw("Uncorrect matrix length");
    }*/
  inputConnections=connections;
}

void Layer::setOutputConnections(list<Connection*> connections){
  /*  if(connections[0].length()!=numUnits){
    throw("Uncorrect matrix height");
    }*/
  outputConnections=connections;
}

void Layer::addInputConnections(Connection* _connection){
  inputConnections.push_back(_connection);
}

void Layer::addOutputConnections(Connection* _connection){
  outputConnections.push_back(_connection);
}

realv Layer::signalWeighting(FeatureVector signal, Mat weights){
  if(signal.getLength()!=((uint)weights.cols)){
    throw length_error("Uncorrect length between signal and weights");
  }
  realv sum=0;
  for(uint i=0;i<signal.getLength();i++){
    sum+=signal[0]*weights.at<realv>(0,i);
  }
  return sum;
}

Layer::~Layer(){

}
