/*!
 * \file PBDNN.cpp
 * Body of the PBDNN class.
 * \author Luc Mioulet
 */

#include "PBDNN.hpp"

using namespace std;
using namespace cv;

PBDNN::PBDNN(vector<NeuralNetwork*> _forwards) : forwardPopulation(_forwards), errors(vector<FeatureVector>()) {

}

PBDNN::PBDNN(uint _numNetworks, uint _numEntries, uint _hiddenLayerSize, ValueVector _mean, ValueVector _stdDev) : forwardPopulation(vector<NeuralNetwork*>()) , errors(vector<FeatureVector>()) {
  int random1;
  int random2;
    for(uint i=0;i<_numNetworks;i++){
      InputLayer* il = new InputLayer(_numEntries, _mean, _stdDev);
      LayerSigmoid* th = new LayerSigmoid(_hiddenLayerSize);
      LayerSigmoid* out = new LayerSigmoid(_numEntries);
      Connection* c1 = new Connection(il,th,i+random1);
      Connection* c2 = new Connection(th,out,_numEntries+i+random2);
      vector<Layer*> layers;
      layers.push_back(il);
      layers.push_back(th);
      layers.push_back(out);
      vector<Connection*> connections;
      connections.push_back(c1);
      connections.push_back(c2);
      NeuralNetwork network(il,layers,out,connections,"network");
      forwardPopulation.push_back(network.clone());
    }
}

void PBDNN::forwardSequence(std::vector<FeatureVector> _sequence){
  MSEMeasurer mse;
  uint seqSize=_sequence.size();
  errors = vector<FeatureVector>(seqSize,FeatureVector(forwardPopulation.size()));
  for(uint i=0;i<seqSize;i++){
    for(uint j=0;j<forwardPopulation.size();j++){
      forwardPopulation[j]->forward(_sequence[i]);
      errors[i][j]=mse.totalError(_sequence[i],forwardPopulation[j]->getOutputSignal());
    }
  }
}

vector<NeuralNetwork*> PBDNN::getPopulation() const{
  return forwardPopulation;
}

vector<FeatureVector> PBDNN::getOutputSequence(){
  if(errors.size()<=0){
    throw logic_error("PBDNN : no sequence forwarded");
  }
  return errors;
}

PBDNN::~PBDNN(){

}
