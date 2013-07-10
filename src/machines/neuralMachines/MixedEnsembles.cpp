/*!
 * \file MixedEnsembles.cpp
 * Body of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include "MixedEnsembles.hpp"

using namespace std;
using namespace cv;

MixedEnsembles::MixedEnsembles(std::vector<NeuralNetworkPtr> _networks,std::vector<ImageFrameExtractor> _ifes, std::vector<uint> _linkedToIFE, Connector _connector, NeuralNetworkPtr _outputNetwork) : networks(_networks), ifes(_ifes), linkedToIFE(_linkedToIFE),outputNetwork(_outputNetwork){
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
  FeatureVector connectedOutput = getConnectorOutput(_matrix,_i);
  outputNetwork->forward(connectedOutput);
}

FeatureVector MixedEnsembles::getConnectorOutput(Mat _matrix, uint _i){
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
    return connectedOutput;
}

NeuralNetworkPtr MixedEnsembles::getOutputNetwork(){
  return outputNetwork;
}

void MixedEnsembles::setOutputNetwork(NeuralNetworkPtr _outputNet){
  outputNetwork = _outputNet;
}
/*
vector<NeuralNetworkPtr> MixedEnsembles::getNetworks(){
  return networks;
}

void MixedEnsembles::setNetworks(NeuralNetworkPtr _networks){
  networks = _networks;
}

vector<ImageFrameExtractor> MixedEnsembles::getIFEs(){
  return ifes;
}

void MixedEnsembles::setIFEe(vector<ImageFrameExtractor> _ifes){
  ifes=_ifes;
}

vector<uint> MixedEnsembles::getLinkedToIFE(){
  return linkedToIFE;
}

void MixedEnsembles::setLinkedToIFE(){

}*/


MixedEnsembles::~MixedEnsembles(){

}

/*ofstream& operator<<(ofstream& _ofs, const MixedEnsembles& _pop){
  _ofs << " < ";
  vector<NeuralNetworkPtr> population= _pop.getPopulation();
  _ofs << population.size() << endl;
  for(uint i = 0; i<population.size();i++){
    _ofs << *(population[i].get()) << endl;
  }
  _ofs << " > ";
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, PBDNN& _pop){
  string temp;
  int popSize;
  vector<NeuralNetworkPtr> population = vector<NeuralNetworkPtr>();
  _ifs >> temp;
  _ifs >> popSize;
  for(uint i = 0; i<popSize;i++){
    NeuralNetwork nnTemp;
    _ifs >> nnTemp;
    population.push_back(NeuralNetworkPtr(new NeuralNetwork(nnTemp)));
  }
  _pop = PBDNN(population);
  _ifs >> temp;
  return _ifs;
  }*/
