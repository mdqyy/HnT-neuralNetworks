/*!
 * \file MixedEnsembles.cpp
 * Body of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include "MixedEnsembles.hpp"

using namespace std;
using namespace cv;

MixedEnsembles::MixedEnsembles() : networks(vector<NeuralNetworkPtr>()),ifes(vector<ImageFrameExtractor>()), linkedToIFE(vector<uint>()),outputNetwork(NeuralNetworkPtr()){
  
}

MixedEnsembles::MixedEnsembles(vector<NeuralNetworkPtr> _networks,vector<ImageFrameExtractor> _ifes, vector<uint> _linkedToIFE, NeuralNetworkPtr _outputNetwork) : networks(_networks), ifes(_ifes), linkedToIFE(_linkedToIFE),outputNetwork(_outputNetwork){
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

NeuralNetworkPtr MixedEnsembles::getOutputNetwork() const{
  return outputNetwork;
}

void MixedEnsembles::setOutputNetwork(NeuralNetworkPtr _outputNet){
  outputNetwork = _outputNet;
}

vector<NeuralNetworkPtr> MixedEnsembles::getNetworks() const{
  return networks;
}

void MixedEnsembles::setNetworks(vector<NeuralNetworkPtr> _networks){
  networks = _networks;
}

vector<ImageFrameExtractor> MixedEnsembles::getIFEs() const{
  return ifes;
}

void MixedEnsembles::setIFEe(vector<ImageFrameExtractor> _ifes){
  ifes=_ifes;
}

vector<uint> MixedEnsembles::getLinkedToIFE() const{
  return linkedToIFE;
}

void MixedEnsembles::setLinkedToIFE(vector<uint> _linkedToIFE){
  linkedToIFE = _linkedToIFE;
}


MixedEnsembles::~MixedEnsembles(){

}

ofstream& operator<<(ofstream& _ofs, const MixedEnsembles& _ensemble){
  _ofs << " < ";
  vector<NeuralNetworkPtr> inputNets = _ensemble.getNetworks();
  vector<uint> links = _ensemble.getLinkedToIFE();
  vector<ImageFrameExtractor> ifes = _ensemble.getIFEs();
  _ofs << inputNets.size() << endl;
  for(uint i=0;i<inputNets.size();i++){
    _ofs << *(inputNets[i].get())<< endl;
  }
  _ofs << links.size() << endl;
  _ofs <<" [ ";
  for(uint i=0;i<links.size();i++){
    _ofs << links[i] << " " ;
  }
  _ofs <" ] ";
  _ofs << ifes.size() << endl;
  for(uint i=0;i<ifes.size();i++){
    _ofs << ifes[i] << endl;
  }
  _ofs <<  *(_ensemble.getOutputNetwork().get()) << endl;
  _ofs << " > ";
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, MixedEnsembles& _ensemble){
  string temp;
  int count;
  uint link;
  vector<NeuralNetworkPtr> population = vector<NeuralNetworkPtr>();
  vector<uint> links = vector<uint>();
  vector<ImageFrameExtractor> ifes =  vector<ImageFrameExtractor>();
  _ifs >> temp;
  _ifs >> count;
  for(uint i = 0; i<count;i++){
    NeuralNetwork nnTemp;
    _ifs >> nnTemp;
    population.push_back(NeuralNetworkPtr(new NeuralNetwork(nnTemp)));
  }
  _ifs >> count;
  _ifs >> temp;
  for(uint i= 0; i < count; i++){
    _ifs >> link;
    links.push_back(link);
  }
  _ifs >> temp;
  _ifs >> count;
  for(uint i= 0; i < count; i++){
    ImageFrameExtractor ife;
    _ifs >> ife;
    ifes.push_back(ife);
  }
  NeuralNetwork outputNet;
  _ifs >> outputNet;
  NeuralNetworkPtr outputNetPtr(new NeuralNetwork(outputNet));
  _ifs >> temp;
  _ensemble = MixedEnsembles(population,ifes,links,outputNetPtr);
  _ifs >> temp;
  return _ifs;
}
