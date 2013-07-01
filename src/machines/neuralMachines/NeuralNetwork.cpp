/*!
 * \file NeuralNetwork.cpp
 * Body of the NeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"

using namespace std;
using namespace cv;

NeuralNetwork::NeuralNetwork() : NeuralMachine("perceptron"),hiddenLayers(vector<LayerPtr>()), connections(vector<ConnectionPtr>()), readForward(true){

}

NeuralNetwork::NeuralNetwork(vector<LayerPtr> _hidden, vector<ConnectionPtr> _connections, bool _forward,string _name) : NeuralMachine(_name), hiddenLayers(_hidden), connections(_connections), readForward(_forward){
  
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& _cnn) : NeuralMachine(_cnn.getName()),hiddenLayers(vector<LayerPtr>()), connections(vector<ConnectionPtr>()), readForward(_cnn.isForward()){
  vector<LayerPtr> tempLayers = _cnn.getHiddenLayers();
  for(uint i=0;i<tempLayers.size();i++){
    hiddenLayers.push_back(LayerPtr(tempLayers[i]->clone()));
  }
  vector<ConnectionPtr> tempConnections = _cnn.getConnections();
  for (uint i=0;i<tempConnections.size();i++){
    connections.push_back(ConnectionPtr(tempConnections[i]));
    connections[i]->setInputLayer(hiddenLayers[i].get());
    hiddenLayers[i]->setOutputConnection(connections[i].get());
    connections[i]->setOutputLayer(hiddenLayers[i+1].get());
    hiddenLayers[i+1]->setInputConnection(connections[i].get());
  }
}



NeuralNetwork* NeuralNetwork::clone() const{
  return new NeuralNetwork(*this);
}

Layer* NeuralNetwork::getInputLayer() const{
  return hiddenLayers[0].get();
}

vector<LayerPtr> NeuralNetwork::getHiddenLayers() const{
  return hiddenLayers;
}

vector<ConnectionPtr> NeuralNetwork::getConnections() const{
  return connections;
}

Layer* NeuralNetwork::getOutputLayer() const{
  return hiddenLayers[hiddenLayers.size()-1].get();
}

FeatureVector NeuralNetwork::getInputSignal() const{
  return  getInputLayer()->createInputSignal();
}

FeatureVector NeuralNetwork::getOutputSignal() const{
  FeatureVector outputSig(getOutputLayer()->getNumUnits());
  FeatureVector temp = getOutputLayer()->getOutputSignal();
  for(uint i=0;i<outputSig.getLength();i++){
    outputSig[i]=temp[i];
  }
  return  outputSig;
}

bool NeuralNetwork::isForward() const{
  return readForward;
}

void NeuralNetwork::setHiddenLayers(std::vector<LayerPtr> _hidden){
  hiddenLayers=_hidden;
}

void NeuralNetwork::forwardSequence(std::vector<FeatureVector> _sequence){
  if(readForward){
    for(uint i=0;i<_sequence.size();i++){
      forward(_sequence[i]);
    }
  }
  else{
    for(int i=_sequence.size()-1;i>=0;i--){
      forward(_sequence[i]);
    }
  }
}

void NeuralNetwork::forward(FeatureVector _signal){
  getInputLayer()->forward(_signal);
}

void NeuralNetwork::suppressLastLayer(){
  hiddenLayers.pop_back();
  connections.pop_back();
  hiddenLayers[hiddenLayers.size()-1]->setOutputConnection(0);
}

void NeuralNetwork::print(ostream& _os) const{
  int numWeights = 0;
  for(uint i=0;i<connections.size();i++){
    numWeights += connections[i]->getWeights().cols*connections[i]->getWeights().rows;
  }
  _os << "Neural network " <<getName()<< " with "<< hiddenLayers.size()<<" and " << numWeights<<" weights."<<endl;
}

NeuralNetwork::~NeuralNetwork(){

}

ofstream& operator<<(ofstream& ofs, const NeuralNetwork& nn){
  ofs << " < ";
  ofs << nn.getName()<< " ";
  ofs << nn.isForward()<< " ";
  vector<ConnectionPtr> connections = nn.getConnections();
  vector<LayerPtr> layers = nn.getHiddenLayers();  
  ofs << layers.size()<< " " << connections.size()<< endl;
  for(uint i=0;i<connections.size();i++){
    ofs << *(connections[i].get());
    ofs << endl;
  }
  for(uint i=0;i<layers.size();i++){
    switch(layers[i]->getLayerType()){
    case 1 :{
      InputLayer* tempIn = (InputLayer*)((layers[i].get()));
      ofs << layers[i]->getLayerType() <<" ";
      ofs << *tempIn;      
      break;}
    case 2 :{
      LayerSigmoid* tempSig = (LayerSigmoid*)((layers[i].get()));
      ofs << layers[i]->getLayerType() <<" ";
      ofs << *tempSig;      
      break;}
    case 3 :{
      LayerTanh* tempTanh = (LayerTanh*)((layers[i].get()));
      ofs << layers[i]->getLayerType() <<" ";
      ofs << *tempTanh;      
      break;}
    case 4 :{
      LayerSoftMax* tempSM = (LayerSoftMax*)((layers[i].get()));
      ofs << layers[i]->getLayerType() <<" ";
      ofs << *tempSM;      
      break;}
    default :{
      break;}
    }
    ofs << endl;
    }
  ofs << " > "<< endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, NeuralNetwork& nn){
  string name, temp;
  int forwardInt, numConnections, numLayers, layerType;
  bool forwardBool;
  vector<ConnectionPtr> connections = vector<ConnectionPtr>();
  vector<LayerPtr> layers = vector<LayerPtr>();
  ifs >> temp ;
  ifs >> name;
  ifs >> forwardInt;
  forwardBool = (forwardInt==1);
  ifs >> numLayers;
  ifs >> numConnections;
  for(int i=0;i<numConnections;i++){
    Connection tempCo;
    ifs >> tempCo;
    ConnectionPtr temp = ConnectionPtr(new Connection(tempCo));
    connections.push_back(temp);
  }
  for(int i=0;i<numLayers;i++){
    ifs >> layerType;
    switch(layerType){
    case 1 :{
      InputLayer tempLayer;
      ifs >> tempLayer;
      LayerPtr temp = LayerPtr(new InputLayer(tempLayer));
      layers.push_back(temp);
      break;
    }
    case 2 :{
      LayerSigmoid tempLayer;
      ifs >> tempLayer;
      LayerPtr temp = LayerPtr(new LayerSigmoid(tempLayer));
      layers.push_back(temp);
      break;
    }
    case 3 :{
      LayerTanh tempLayer;
      ifs >> tempLayer;
      LayerPtr temp = LayerPtr(new LayerTanh(tempLayer));
      layers.push_back(temp);
      break;
    }
    case 4 :{
      LayerSoftMax tempLayer;
      ifs >> tempLayer;
      LayerPtr temp = LayerPtr(new LayerSoftMax(tempLayer));
      layers.push_back(temp);
      break;
    }
    default :{
      break;
    }
    }
  }  
  for(uint i=0;i<numConnections;i++){
    connections[i].get()->setInputLayer(layers[i].get());
    layers[i]->setOutputConnection(connections[i].get());
    connections[i]->setOutputLayer(layers[i+1].get());
    layers[i+1]->setInputConnection(connections[i].get());
  }
  ifs >> temp;
  nn = NeuralNetwork(layers,connections,forwardBool,name);
  return ifs;
}
