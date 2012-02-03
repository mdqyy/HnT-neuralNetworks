/*!
 * \file LayerTanh.cpp
 * Body of the LayerTanh class.
 * \author Luc Mioulet
 */

#include "LayerTanh.hpp"
#include <math.h>

using namespace std;
using namespace cv;

LayerTanh::LayerTanh(uint _numUnits,string _name) : Layer(_numUnits,_name){

}

void LayerTanh::forward(){
  list<Connection*>::iterator it;
  outputSignal.reset(0.0);
  /* Accumulate neuron sum */
  for(it=inputConnections.begin();it!=inputConnections.end();it++){
    FeatureVector layerInputSignal=((*it)->getInputLayer()).getOutputSignal();
    for(uint i=0;i<outputSignal.getLength();i++){
      outputSignal[i]+=signalWeighting(layerInputSignal, (*it)->getWeightsNeuron(i));
    }
  }
  /* Tanh the sum */
  for(uint i=0;i<outputSignal.getLength();i++){
    outputSignal[i]=tanh(outputSignal[i]);

  }
}

void LayerTanh::backward(ErrorVector deltas){

}

LayerTanh::~LayerTanh(){

}


ostream& operator<<(ostream& os, const LayerTanh& l){
  os << "Tanh neuron layer : " << endl;
  os << "\t -Name :"<< l.getName() << endl;
  os << "\t -Units : "<<l.getNumUnits() << endl;
  os << "\t -Input Connections objects : " << l.getInputConnections().size() <<endl;
  os << "\t -Ouput Connections objects : "<< l.getOutputConnections().size() << endl;
  return os;
}
