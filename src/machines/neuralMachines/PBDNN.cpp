/*!
 * \file PBDNN.cpp
 * Body of the PBDNN class.
 * \author Luc Mioulet
 */

#include "PBDNN.hpp"

using namespace std;
using namespace cv;

PBDNN::PBDNN(vector<NeuralNetworkPtr> _forwards) : forwardPopulation(_forwards), errors(vector<FeatureVector>()) {

}

PBDNN::PBDNN(uint _numNetworks, uint _numEntries, uint _hiddenLayerSize, ValueVector _mean, ValueVector _stdDev) : forwardPopulation(vector<NeuralNetworkPtr>()) , errors(vector<FeatureVector>()) {
    for(uint i=0;i<_numNetworks;i++){
      LayerPtr il = LayerPtr(new InputLayer(_numEntries, _mean, _stdDev));
      LayerPtr th = LayerPtr(new LayerSigmoid(_hiddenLayerSize));
      LayerPtr out = LayerPtr(new LayerSigmoid(_numEntries));
      ConnectionPtr c1 = ConnectionPtr(new Connection(il.get(),th.get(),i));
      ConnectionPtr c2 = ConnectionPtr(new Connection(th.get(),out.get(),_numEntries+i));
      vector<LayerPtr> layers;
      layers.push_back(il);
      layers.push_back(th);
      layers.push_back(out);
      vector<ConnectionPtr> connections;
      connections.push_back(c1);
      connections.push_back(c2);
      NeuralNetwork network(layers,connections,"network");
      forwardPopulation.push_back(NeuralNetworkPtr(network.clone()));
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

vector<NeuralNetworkPtr> PBDNN::getPopulation() const{
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
