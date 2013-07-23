/*!
 * \file ImagePopulationInverseTrainer.cpp
 * Body of the ImagePopulationInverseTrainer class.
 * \author Luc Mioulet
 */

#include "ImagePopulationInverseTrainer.hpp"
#include "../../errorMeasurers/AEMeasurer.hpp"

using namespace cv;
using namespace std;

/*!
 * Thread function forwarding data and extracting error.
 */
ErrorVector calculateDeltasTiedWeights(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < delta.getLength(); i++) {
    delta[i] = _derivatives[i]
        * _layer->errorWeighting(_previousLayerDelta,
            _layer->getOutputConnection()->getWeightsFromNeuron(i));
  }
  return delta;
}

ErrorVector calculateOutputDeltasTiedWeights(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < _target.getLength(); i++) {
    //delta[i] = _derivatives[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with MSE
    delta[i] = _layer->getOutputSignal()[i]
        * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with Cross-Entropy
  }
  return delta;
}

void updateConnectionTiedWeights(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate) {
  Mat weights = _connection->getWeights();
  for (int i = 0; i < weights.rows; i++) {
    for (int j = 0; j < weights.cols; j++) {
      weights.at<realv>(i, j) = weights.at<realv>(i, j)
          + _learningRate * _deltas[i]
              * _connection->getInputLayer()->getOutputSignal()[j];
    }
  }
  _connection->setWeights(weights.clone());
}

void backwardTiedWeights(NeuralNetworkPtr _neuralNet, FeatureVector _target, realv _learningRate) {
  vector<ConnectionPtr> connections = _neuralNet->getConnections();
  vector<LayerPtr> layers = _neuralNet->getHiddenLayers();
  vector<ErrorVector> deltas = vector<ErrorVector>();/* pushed in inversed order so be careful */
  for (uint i = layers.size() - 1; i > 0; i--) {
    ValueVector derivatives = layers[i]->getDerivatives();
    if (i == layers.size() - 1) {
      deltas.push_back(
          calculateOutputDeltasTiedWeights(layers[i], _target, derivatives));
    }
    else {
      deltas.push_back(
          calculateDeltasTiedWeights(layers[i], _target, derivatives,
              deltas[deltas.size() - 1]));
    }
  }
  for (uint i = 0; i < connections.size(); i++) {
    updateConnectionTiedWeights(connections[i],
        deltas[connections.size() - i - 1], _learningRate);
  }
  /* tied weights */
  Mat ts = connections[1]->getWeights();
  Mat td = connections[0]->getWeights();
  for (int i = 0; i < ts.rows - 1; i++) {
    for (int j = i; j < td.rows - 1; j++) {
      td.at<realv>(j, i) = ts.at<realv>(i, j);
    }
  }
  connections[0]->setWeights(td);
}

// Thread process per network only forward to find min error networks
void threadImageForwardPerNetwork(vector<NeuralNetworkPtr>* _neuralNets, uint _k, ImageDataset* _trainingData, uint _numberOfElementsToProcess, vector<
    uint>* _indexOrderSelection, vector<vector<realv> > *_errors) {
  uint index = 0;
  AEMeasurer mae;
  for (uint i = 0; i < _numberOfElementsToProcess; i++) {
    index = (*_indexOrderSelection)[i];
    FeatureVector fv = (*_trainingData).getFeatures(index)[0];
    (*_neuralNets)[_k]->forward(fv);
    mae.processErrors((*_neuralNets)[_k]->getOutputSignal(),
        fv);
    (*_errors)[_k][i] = mae.getError();
  }
}

FeatureVector randomSwap(FeatureVector _vec, realv _noise) {
  RNG random((uint) getTickCount());
  FeatureVector result(_vec.getLength());
  realv val;
  if (_noise > 0.0) {
    for (uint i = 0; i < _vec.getLength(); i++) {
      random.next();
      val = random.uniform(0.0, 1.0);
      if (val < _noise) {
        result[i] = abs(_vec[i] - 1);
      }
      else {
        result[i] = _vec[i];
      }
    }
  }
  else {
    result = _vec;
  }
  return result;
}

// Thread process per network forward and backward to train min error networks
void threadImageForwardBackwardPerNetwork(vector<NeuralNetworkPtr>* _neuralNets, uint _k, ImageDataset* _imageDataset, vector<
    vector<uint> >* _learningAffectations, realv _learningRate, uint _maxTrained, realv _noise) {
  FeatureVector blackTarget = FeatureVector(_imageDataset->getFeatures(0)[0]);
  RNG randomK((uint) getTickCount());
  uint index = 0;
  FeatureVector input, target;
  for (uint i = 0; i < _maxTrained && i < (*_learningAffectations)[_k].size();
      i++) {
    /* first backward random bad sample*/
    randomK.next();
    uint randI = 0;
    for (uint j = 0; j < (*_neuralNets).size() - 1; j++) {
      if ((*_learningAffectations)[j].size() > 0 && j != _k) {
        randomK.next();
        randI = randomK.uniform(0, (*_learningAffectations)[j].size());
        index = (*_learningAffectations)[j][randI];
        input = randomSwap((*_imageDataset).getFeatures(index)[0], _noise);
        target = (*_imageDataset).getFeatures(index)[0];
        (*_neuralNets)[_k]->forward(input);
        for (uint v = 0; v < target.getLength(); v++) {
          if (target[v] == 0.0) {
            blackTarget[v] = 1.0;
          }
          else {
            blackTarget[v] = 0.0;
          }

        }
      }
    }
    backwardTiedWeights((*_neuralNets)[_k], blackTarget, _learningRate);
    /* forward backward good sample */
    index = (*_learningAffectations)[_k][i];
    input = randomSwap((*_imageDataset).getFeatures(index)[0], _noise);
    target = (*_imageDataset).getFeatures(index)[0];
    (*_neuralNets)[_k]->forward(input);
    backwardTiedWeights((*_neuralNets)[_k], target, _learningRate);
  }
}

/* Population trainer methods */
ImagePopulationInverseTrainer::ImagePopulationInverseTrainer(PBDNN& _population, ImageDataset& _trainingDataset, ImageDataset& _validationDataset, LearningParams& _params, ostream& _log) :
    population(_population), trainingDataset(_trainingDataset), validationDataset(
        _validationDataset), params(_params), log(_log), endurance(vector<uint>(_population.getPopulation().size(), _params.getDodges())) {

}

FeatureVector ImagePopulationInverseTrainer::noiseTarget(FeatureVector _vec) {
  RNG random((uint) getTickCount());
  FeatureVector result(_vec.getLength());
  realv val;
  if (params.getNoise() > 0.0) {
    for (uint i = 0; i < _vec.getLength(); i++) {
      random.next();
      val = random.uniform(0.0, 1.0);
      if (val < params.getNoise()) {
        result[i] = abs(_vec[i] - 1);
      }
      else {
        result[i] = _vec[i];
      }
    }
  }
  else {
    result = _vec;
  }
  return result;
}

vector<uint> ImagePopulationInverseTrainer::defineIndexOrderSelection(uint _numSequences) {
  vector<uint> indexOrder;
  for (uint i = 0; i < _numSequences; i++) {
    indexOrder.push_back(i);
  }
  int exchangeIndex = 0;
  RNG random(getTickCount());
  random.next();
  for (uint i = 0; i < _numSequences; i++) {
    random.next();
    exchangeIndex = random.uniform(0, _numSequences);
    swap(indexOrder[i], indexOrder[exchangeIndex]);
  }
  return indexOrder;
}

void ImagePopulationInverseTrainer::train() {
  uint i = params.getActualIteration();
  do {
    i++;
    log << "Iteration" << i << endl;
    trainOneIteration();
    params.setActualIteration(i);
    params.setLearningRate(
        params.getLearningRate() * params.getLearningRateDecrease());
    params.setErrorToFirst(
        params.getErrorToFirst() * params.getErrorToFirstIncrease());
    if (params.isSavedDuringProcess()) {
      ostringstream name;
      name << params.getSaveLocation() << "/populationIteration" << i << ".txt";
      ofstream outStream(name.str().c_str());
      outStream << this->population;
      outStream << this->params;
      outStream.close();
    }
    if (params.isValidatedDuringProcess()
        && (i + 1) % params.getValidateEveryNIteration() == 0) {
      log << "Validation " << endl;
      validateIteration();
    }
  } while (i < params.getMaxIterations());
}

void ImagePopulationInverseTrainer::trainOneIteration() {
  vector<uint> indexOrderSelection = defineIndexOrderSelection(
      trainingDataset.getNumberOfImages());
  uint numberOfElementsToProcess = trainingDataset.getNumberOfImages()
      * params.getMaxTrainedPercentage();
  vector<NeuralNetworkPtr> neuralNets = population.getPopulation();
  AEMeasurer mae;
  uint maxTrained = numberOfElementsToProcess / neuralNets.size();
  vector<vector<realv> > errors;
  for (uint k = 0; k < neuralNets.size(); k++) {
    errors.push_back(vector<realv>(numberOfElementsToProcess, 10e+9));
  }
  /* Use threaded forward process per network */
  vector<boost::thread *> threadsForward;
  for (uint k = 0; k < neuralNets.size(); k++) {
    threadsForward.push_back(
        new boost::thread(threadImageForwardPerNetwork, &neuralNets, k,
            &trainingDataset, numberOfElementsToProcess, &indexOrderSelection,
            &errors));
  }
  for (uint k = 0; k < neuralNets.size(); k++) {
    threadsForward[k]->join();
    delete threadsForward[k];
  }

  /* Find learning affectations, regenerate if endurance is 0*/
  realv maxError = params.getProximity()
      * ((realv) neuralNets[0]->getInputLayer()->getNumUnits());
  vector<vector<uint> > learningAffectations = determineLearningAffectations(
      errors, indexOrderSelection, numberOfElementsToProcess, maxError);
  for (uint k = 0; k < neuralNets.size(); k++) {
    log << "Network " << k << " obtained " << learningAffectations[k].size() << " samples" << endl;
    if (learningAffectations[k].size() == 0) {
      endurance[k] = endurance[k] - 1;
      if (endurance[k] == 0) {
        endurance[k] = params.getDodges();
        population.regenerate(k);
      }
    }
    else {
      endurance[k] = params.getDodges();
    }
  }
  /*Train according to learning affectations*/
  vector<boost::thread *> threadsForwardBackward;
  for (uint k = 0; k < neuralNets.size(); k++) {
    threadsForwardBackward.push_back(
        new boost::thread(threadImageForwardBackwardPerNetwork, &neuralNets, k,
            &trainingDataset, &learningAffectations, params.getLearningRate(),
            maxTrained, params.getNoise()));
  }
  for (uint k = 0; k < neuralNets.size(); k++) {
    threadsForwardBackward[k]->join();
    delete threadsForwardBackward[k];
  }
}

void ImagePopulationInverseTrainer::validateIteration() {
  vector<uint> indexOrderSelection = defineIndexOrderSelection(
      validationDataset.getNumberOfImages());
  uint numberOfElementsToProcess = validationDataset.getNumberOfImages()
      * params.getMaxTrainedPercentage();
  vector<NeuralNetworkPtr> neuralNets = population.getPopulation();
  AEMeasurer mae;
  vector<vector<realv> > errors;
  for (uint k = 0; k < neuralNets.size(); k++) {
    errors.push_back(vector<realv>(numberOfElementsToProcess, 10e+9));
  }
  /* Use threaded forward process per network */
  vector<boost::thread *> threadsForward;
  for (uint k = 0; k < neuralNets.size(); k++) {
    threadsForward.push_back(
        new boost::thread(threadImageForwardPerNetwork, &neuralNets, k,
            &validationDataset, numberOfElementsToProcess, &indexOrderSelection,
            &errors));
  }
  for (uint k = 0; k < neuralNets.size(); k++) {
    threadsForward[k]->join();
    delete threadsForward[k];
  }
  /* Find learning affectations, regenerate if endurance is 0*/
  realv maxError = params.getProximity()
      * ((realv) neuralNets[0]->getInputLayer()->getNumUnits());
  vector<vector<uint> > learningAffectations = determineLearningAffectations(
      errors, indexOrderSelection, numberOfElementsToProcess, maxError);
  vector<realv> netsError = vector<realv>(neuralNets.size(),0.0);
  log << "network |\t Best errors \t| \t timesSelected" << endl;
  for(uint k = 0;k < neuralNets.size();k++){
    for(uint i = 0; i < learningAffectations[k].size(); i++){
      netsError[k] += errors[k][i];
    }
    log << k << " \t | \t " << netsError[k]/((realv)learningAffectations[k].size()) << "\t | \t" << learningAffectations[k].size() << endl;
  }
}

ImagePopulationInverseTrainer::~ImagePopulationInverseTrainer() {

}

vector<vector<uint> > ImagePopulationInverseTrainer::determineLearningAffectations(vector<
    vector<realv> >& _errors, vector<uint>& _indexOrderSelection, uint _numberOfElementsToProcess, realv _maxError) {
  vector<vector<uint> > learningAffectations = vector<vector<uint> >();
  for (uint k = 0; k < _errors.size(); k++) {
    learningAffectations.push_back(vector<uint>());
  }
  uint bestNetwork = 0;
  realv minError = 10e+9;
  for (uint j = 0; j < _numberOfElementsToProcess; j++) {
    bestNetwork = 0;
    minError = 10e+9;
    for (uint k = 0; k < _errors.size(); k++) {
      if (_errors[k][j] < minError) {
        minError = _errors[k][j];
        bestNetwork = k;
      }
    }
    if (minError < _maxError) {
      learningAffectations[bestNetwork].push_back(_indexOrderSelection[j]);
    }
  }
  return learningAffectations;
}

