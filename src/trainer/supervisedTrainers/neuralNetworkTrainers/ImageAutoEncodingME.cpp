/*!
 * \file ImageAutoEncodingME.cpp
 * Body of the ImageAutoEncodingME class.
 * \author Luc Mioulet
 */

#include "ImageAutoEncodingME.hpp"

using namespace std;
using namespace cv;


ImageAutoEncodingME::ImageAutoEncodingME(MixedEnsembles& _machine, ImageDataset& _dataset, ImageDataset& _testDataset, LearningParams _params, std::ostream& _log) : machine(_machine),dataset(_dataset), testDataset(_testDataset),params(_params), log(_log){

}

void ImageAutoEncodingME::train(){
  uint i = params.getActualIteration();
  do {
    i++;
    log << "Iteration" << i << endl;
    trainOneIteration();
    params.setActualIteration(i);
    params.setLearningRate(params.getLearningRate() * params.getLearningRateDecrease());
    if (params.isSavedDuringProcess()) {
      ostringstream name;
      name << params.getSaveLocation() << "/mixedEnsembleIteration" << i << ".txt";
      ofstream outStream(name.str().c_str());
      outStream << this->machine;
      outStream << this->params;
      outStream.close();
    }
    if (params.isValidatedDuringProcess() && (i+1)%params.getValidateEveryNIteration()==0) {
      log << "Validation at step "<< i << endl;
      validateIteration();
    }
  } while (i < params.getMaxIterations());
}

void ImageAutoEncodingME::trainOneIteration(){
  vector<uint> indexOrderSelection = defineIndexOrderSelection(dataset.getNumberOfImages());
  uint numberOfElementsToProcess = dataset.getNumberOfImages()*params.getMaxTrainedPercentage();
  AEMeasurer mae;
  uint index = 0;
  FeatureVector input,target;
  NeuralNetworkPtr netPtr = machine.getOutputNetwork();
  for(uint i = 0;i<numberOfElementsToProcess;i++){
    index = indexOrderSelection[i];
    Mat image= dataset.getMatrix(index,0);
    for(uint j = 0; j< image.rows;j++){
      //machine.forward(input,j);
	/*	error+=measureSampleError(machine.getOutputSignal(),target);*/
      FeatureVector input =  machine.getConnectorOutput(image,j);
      FeatureVector target = noiseTarget(input);
      netPtr->forward(input);
      backward(target); 
    }
  }
}

FeatureVector ImageAutoEncodingME::noiseTarget(FeatureVector _vec){
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

vector<uint> ImageAutoEncodingME::defineIndexOrderSelection(uint _numSequences){
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

ErrorVector ImageAutoEncodingME::calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < delta.getLength(); i++) {
    delta[i] = _derivatives[i] * _layer->errorWeighting(_previousLayerDelta, _layer->getOutputConnection()->getWeightsFromNeuron(i));
  }
  return delta;
}

ErrorVector ImageAutoEncodingME::calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < _target.getLength(); i++) {
    delta[i] = _derivatives[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with MSE
    delta[i] =  _layer->getOutputSignal()[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with Cross-Entropy
  }
  return delta;
}

void ImageAutoEncodingME::updateConnection(ConnectionPtr _connection, ErrorVector _deltas) {
  Mat weights = _connection->getWeights();
  for (int i = 0; i < weights.rows; i++) {
    for (int j = 0; j < weights.cols; j++) {
      weights.at<realv>(i, j) = weights.at<realv>(i, j) + params.getLearningRate() * _deltas[i] * _connection->getInputLayer()->getOutputSignal()[j];
    }
  }
  _connection->setWeights(weights.clone());
}

void ImageAutoEncodingME::backward(FeatureVector _target) {
  NeuralNetworkPtr outputNet = machine.getOutputNetwork();
  vector<ConnectionPtr> connections = outputNet->getConnections();
  vector<LayerPtr> layers = outputNet->getHiddenLayers();
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

void ImageAutoEncodingME::validateIteration(){
  AEMeasurer ae;
  uint testSampleSize = testDataset.getNumberOfImages()*params.getMaxTrainedPercentage();
  vector<uint> order = defineIndexOrderSelection(testDataset.getNumberOfImages());
  uint index = 0;
  realv totalError = 0;
  uint totalNumberOfFrames = 0;
  NeuralNetworkPtr netPtr = machine.getOutputNetwork();
  for(uint i = 0; i < testSampleSize; i++){
    index = order[i];
    Mat image = testDataset.getMatrix(index,0);
    for(uint j = 0; j< image.cols; j=j+6){
      FeatureVector intermediateOutput = machine.getConnectorOutput(image,j);
      netPtr->forward(intermediateOutput);
      FeatureVector output = machine.getOutput();
      ae.processErrors(output,intermediateOutput);
      totalError += ae.getError();
      totalNumberOfFrames ++;
    }
    log << "Output Error is : " << totalError/((realv)totalNumberOfFrames)<<endl;
  }
}

ImageAutoEncodingME::~ImageAutoEncodingME(){

}
