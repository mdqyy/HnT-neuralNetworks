/*!
 * \file MixedEnsembles.cpp
 * Body of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include "MixedEnsembles.hpp"

using namespace std;
using namespace cv;

MixedEnsembles::MixedEnsembles(std::vector<NeuralNetworkPtr> _networks,std::vector<ImageFrameExtractor> _ifes, std::vector<uint> _linkedToIFE, Connector _connector, NeuralNetwork _outputNetwork) : networks(_networks), ifes(_ifes), linkedToIFE(_linkedToIFE),outputNetwork(_outputNetwork){
  if(networks.size()!=linkedToIFE.size()){
    cerr << "wrong network/ifes size" <<networks.size()<< "/" << ifes.size() << endl;
  }
  for (int i = 0; i < linkedToIFE.size(); i++){ /*Check in all links are good */
    if(linkedToIFE[i]>ifes.size()){
      cerr << "linked to ife "<< i << " : " << linkedToIFE[i] << endl;
    }
  }
  vector<LayerPtr> outputLayers;
  for(uint i = 0 ; i <networks.size();i++){
    outputLayers.push_back(networks[i]->getHiddenLayers().back());
  }
  connector = Connector(outputLayers);
}

void threadForwardPerNetwork(vector<NeuralNetworkPtr>* _neuralNets, uint _k, FeatureVector _fv){
    (*_neuralNets)[_k]->forward(_fv);
}


void MixedEnsembles::forwardMatrix(Mat _matrix){
  uint width = _matrix.cols;
  for (uint i = 0; i < width; ++i){  
    forwardOnPixel(_matrix,i);
  }
}

void MixedEnsembles::forwardOnPixel(Mat _matrix, uint _i){
    vector<FeatureVector> inputs;
    FeatureVector connectedOutput;
    vector<boost::thread * > threadsForward;
    for(uint j=0;j<ifes.size();j++){
      inputs.push_back(ifes[j].getFrameCenteredOn(_matrix,_i));
    }
    uint index = 0;
    for(uint l=0;l<linkedToIFE.size();l++){
      index=linkedToIFE[l];
      networks[l]->forward(inputs[index]);
      threadsForward.push_back(new boost::thread(threadForwardPerNetwork,&networks, l, inputs[index]));
    }
    for(uint n=0; n<networks.size();n++){
      threadsForward[n]->join();
      delete threadsForward[n];
    }
    connectedOutput = connector.concatenateOutputs();
    outputNetwork.forward(connectedOutput);
}


MixedEnsembles::~MixedEnsembles(){

}
