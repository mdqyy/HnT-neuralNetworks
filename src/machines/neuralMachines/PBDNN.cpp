/*!
 * \file PBDNN.cpp
 * Body of the PBDNN class.
 * \author Luc Mioulet
 */

#include "PBDNN.hpp"

using namespace std;
using namespace cv;

PBDNN::PBDNN(vector<NeuralNetwork*>& _forwards) : forwardPopulation(_forwards), errors(vector<FeatureVector>()) {

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

vector<NeuralNetwork*>& PBDNN::getPopulation() const{
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
