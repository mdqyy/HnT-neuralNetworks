/*!
 * \file MixedEnsembles.cpp
 * Body of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include "MixedEnsembles.hpp"

MixedEnsembles::MixedEnsembles(std::vector<NeuralNetworkPtr> _networks,std::vector<ImageFrameExtractor> _ifes, std::vector<int> _linkedToIFE, Connector _connector, NeuralNetwork _outputNetwork) : networks(_networks), ifes(_ifes), linkedToIFE(_linkedToIFE),connector(_networks),outputNetwork(_outputNetwork){
  if(networks.size()!=linkedToIFE.size()){
    cerr << "wrong network/ifes size" <<networks.size()<< "/" << ifes.size() << endl;
  }
  for (int i = 0; i < linkedToIFE; i++){ /*Check in all links are good */
    if(linkedToIFE[i]>ifes.size){
      cerr << "linked to ife "<< i << " : " << linkedToIFE[i] << endl;
    }
  }
}

void forwardMatrix(cv::Mat _matrix){
  uint height = _matrix.cols;
  vector< vector<FeatureVector> > inputs;
  for(uint i=0;i<ifes.size;i++){
    
  }
}


void forwardSequence(std::vector<FeatureVector> _sequence){
  cerr << "Niet Kamarade, does not work here " << endl;
}
  

void forward(FeatureVector _sample){
  cerr << "Niet Kamarade, does not work here " << endl;
}

MixedEnsembles::~MixedEnsembles(){

}
