/*!
 * \file MixedEnsembles.cpp
 * Body of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include "MixedEnsembles.hpp"

MixedEnsembles::MixedEnsembles(std::vector<NeuralNetworkPtr> _networks,std::vector<ImageFrameExtractor> _ifes, Connector _connector, NeuralNetwork _outputNetwork) : networks(_networks), ifes(_ifes), connector(_connector),outputNetwork(_outputNetwork){
  if(networks.size()!=ifes.size()){
    cerr << "wrong network/ifes size" <<networks.size()<< "/" << ifes.size() << endl;
  }
  if(connector){
  }
}

MixedEnsembles::~MixedEnsembles(){

}
