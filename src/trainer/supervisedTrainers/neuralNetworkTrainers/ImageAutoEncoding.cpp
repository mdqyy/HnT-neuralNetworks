/*!
 * \file ImageAutoEncoding.cpp
 * Body of the ImageAutoEncoding class.
 * \author Luc Mioulet
 */

#include "ImageAutoEncoding.hpp"

using namespace std;
using namespace cv;


ImageAutoEncoding::ImageAutoEncoding(Machine& _machine, ImageDataset& _dataset, LearningParams _params, std::ostream& _log) : machine(_machine),dataset(_dataset),params(_params), log(_log){

}

void ImageAutoEncoding::train(){
  uint i = params.getActualIteration();
  do {
    i++;
    log << "Iteration" << i << endl;
    trainOneIteration();
    params.setActualIteration(i);
    params.setLearningRate(params.getLearningRate() * params.getLearningRateDecrease());
    if (params.isSavedDuringProcess()) {
      /*      ostringstream name;
      name << params.getSaveLocation() << "/populationIteration" << i << ".txt";
      ofstream outStream(name.str().c_str());
      outStream << this->population;
      outStream << this->params;
      outStream.close();*/
    }
    if (params.isValidatedDuringProcess() && i%params.getValidateEveryNIteration()==0) {
      log << "Validation "<< endl;
      validateIteration();
    }
  } while (i < params.getMaxIterations());
}

void ImageAutoEncoding::trainOneIteration(){
  vector<uint> indexOrderSelection = defineIndexOrderSelection(dataset.getNumberOfImages());
  uint numberOfElementsToProcess = dataset.getNumberOfImages()*params.getMaxTrainedPercentage();
  AEMeasurer mae;
  uint index = 0;
  FeatureVector input,target;
  for(uint i = 0;i<numberOfElementsToProcess;i++){
    index = indexOrderSelection[i];
    vector<FeatureVector> image= dataset.getFeatures(index);
    for(uint j = 0; j< image.size();j++){
	input = noiseTarget(image[j]);
	target = image[j];
      	machine.forward(input);
	/*	error+=measureSampleError(machine.getOutputSignal(),target);*/
	backward(target); 
    }
  }
}

FeatureVector ImageAutoEncoding::noiseTarget(FeatureVector _vec){
  RNG random((uint) getTickCount());
  FeatureVector result(_vec.getLength());
  realv val;
  if(params.getNoise()>0.0){
    for(uint i=0;i<_vec.getLength();i++){
      random.next();
      val = random.uniform(0.0,1.0);
      if(val < params.getNoise()){
	result[i]=abs(_vec[i]-1);
      }
      else{
	result[i]=_vec[i];
      }
    }
  }
  else{
    result = _vec;
  }
  return result;
}

vector<uint> ImageAutoEncoding::defineIndexOrderSelection(uint _numSequences){
  vector<uint> indexOrder;
  for(uint i=0 ;  i<_numSequences; i++){
    indexOrder.push_back(i);
  }
  int exchangeIndex=0;
  RNG random(getTickCount());
  random.next();
  for(uint i=0 ;  i<_numSequences; i++){	 
    random.next();
    exchangeIndex=random.uniform(0,_numSequences);
    swap(indexOrder[i],indexOrder[exchangeIndex]);
  } 
  return indexOrder;
}

ErrorVector ImageAutoEncoding::calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < delta.getLength(); i++) {
    delta[i] = _derivatives[i] * _layer->errorWeighting(_previousLayerDelta, _layer->getOutputConnection()->getWeightsFromNeuron(i));
  }
  return delta;
}

ErrorVector ImageAutoEncoding::calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < _target.getLength(); i++) {
    delta[i] = _derivatives[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with MSE
    delta[i] =  _layer->getOutputSignal()[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with Cross-Entropy
  }
  return delta;
}

void ImageAutoEncoding::updateConnection(ConnectionPtr _connection, ErrorVector _deltas) {
  Mat weights = _connection->getWeights();
  for (int i = 0; i < weights.rows; i++) {
    for (int j = 0; j < weights.cols; j++) {
      weights.at<realv>(i, j) = weights.at<realv>(i, j) + params.getLearningRate() * _deltas[i] * _connection->getInputLayer()->getOutputSignal()[j];
    }
  }
  _connection->setWeights(weights.clone());
}

void ImageAutoEncoding::backward(FeatureVector _target) {
  vector<ConnectionPtr> connections = machine.getConnections();
  vector<LayerPtr> layers = machine.getHiddenLayers();
  vector<ErrorVector> deltas = vector<ErrorVector>();/* pushed in inversed order so be careful */
  for (uint i = layers.size() - 1; i > 0; i--) {
    ValueVector derivatives = layers[i]->getDerivatives();
    if (i == layers.size() - 1) {
      deltas.push_back(calculateOutputDeltas(layers[i], _target, derivatives));
    } else {
      deltas.push_back(calculateDeltas(layers[i], _target, derivatives, deltas[deltas.size() - 1]));
    }
  }
  for (uint i = 0; i < connections.size(); i++) {
    updateConnection(connections[i], deltas[connections.size() - i - 1]);
  }
  /* tied weights */
  Mat ts = connections[1]->getWeights();
  Mat td = connections[0]->getWeights();
  for(int i=0;i<ts.rows-1;i++){
    for(int j=i;j<td.rows-1;j++){
      td.at<realv>(j,i)=ts.at<realv>(i,j);
    }
  }
  connections[0]->setWeights(td);
}

void validateIteration(){
  /*! TODO */
}

ImageAutoEncoding::~ImageAutoEncoding(){

}
