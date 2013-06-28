/*!
 * \file PopulationInverseTrainer.cpp
 * Body of the PopulationInverseTrainer class.
 * \author Luc Mioulet
 */

#include "PopulationInverseTrainer.hpp"
#include "../../errorMeasurers/AEMeasurer.hpp"

using namespace cv;
using namespace std;


/*!
 * Thread function forwarding data and extracting error.
 */
ErrorVector calculateDeltasTiedWeights(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < delta.getLength(); i++) {
    delta[i] = _derivatives[i] * _layer->errorWeighting(_previousLayerDelta, _layer->getOutputConnection()->getWeightsFromNeuron(i));
  }
  return delta;
}

ErrorVector calculateOutputDeltasTiedWeights(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < _target.getLength(); i++) {
    delta[i] = _derivatives[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with MSE
    delta[i] =  _layer->getOutputSignal()[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer with Cross-Entropy
  }
  return delta;
}

void updateConnectionTiedWeights(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate) {
  Mat weights = _connection->getWeights();
  for (int i = 0; i < weights.rows; i++) {
    for (int j = 0; j < weights.cols; j++) {
      weights.at<realv>(i, j) = weights.at<realv>(i, j) + _learningRate * _deltas[i] * _connection->getInputLayer()->getOutputSignal()[j];
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
      deltas.push_back(calculateOutputDeltasTiedWeights(layers[i], _target, derivatives));
    } else {
      deltas.push_back(calculateDeltasTiedWeights(layers[i], _target, derivatives, deltas[deltas.size() - 1]));
    }
  }
  for (uint i = 0; i < connections.size(); i++) {
    updateConnectionTiedWeights(connections[i], deltas[connections.size() - i - 1], _learningRate);
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


// Thread process per network only forward to find min error networks
void threadForwardPerNetwork(vector<NeuralNetworkPtr>* _neuralNets, uint _k, RegressionDataset* _regData, uint _numberOfElementsToProcess, vector<uint>* _indexOrderSelection, vector<vector<realv> > *_errors){
  uint index = 0;
  AEMeasurer mae;
  for (uint i = 0; i < _numberOfElementsToProcess; i++) {
    index = (*_indexOrderSelection)[i];
    FeatureVector fv = (*_regData)[index][0];
    (*_neuralNets)[_k]->forward(fv);
    mae.processErrors((*_neuralNets)[_k]->getOutputSignal(), (*_regData).getTargetSample(index, 0));
    (*_errors)[_k][i]=mae.getError();
  }
}


// Thread process per network forward and backward to train min error networks
void threadForwardBackwardPerNetwork(vector<NeuralNetworkPtr>* _neuralNets, uint _k, RegressionDataset* _regData, vector<vector<uint> >* _learningAffectations, realv _learningRate,uint _maxTrained){
  FeatureVector blackTarget = FeatureVector(_regData->getTargetSample(0, 0).getLength());
  RNG randomK((uint) getTickCount());
  uint index = 0;
  /* first backward random bad sample, pb ?*/
 /* for(uint i=0; i< _maxTrained && i< (*_learningAffectations)[_k].size(); i++){
    randomK.next();
    uint randK = 0;
    uint randI = 0;
    for(uint i = 0; i < (*_neuralNets).size()-1; i++){
      do {
	randK = randomK.uniform(0, (*_neuralNets).size());
      } while( (randK == _k) && (*_learningAffectations)[randK].size() > 0);
      randomK.next();
      randI = randomK.uniform(0, (*_learningAffectations)[randK].size());
      index = (*_learningAffectations)[randK][randI];
      (*_neuralNets)[_k]->forward((*_regData)[index][0]);
      bitwise_not((*_regData)[index][0].getMat(),inversed);
      blackTarget = FeatureVector(inversed);
      backwardTiedWeights((*_neuralNets)[_k], blackTarget, _learningRate);
    }
  }*/
  /* then backward good samples*/
  for(uint i=0; i< _maxTrained && i< (*_learningAffectations)[_k].size(); i++){
    /* first backward random bad sample*/
    randomK.next();
    uint randK = 0;
    uint randI = 0;
    for(uint i = 0; i < (*_neuralNets).size()-1; i++){
      do {
	randK = randomK.uniform(0, (*_neuralNets).size());
      } while( (randK == _k) && (*_learningAffectations)[randK].size() > 0);
      randomK.next();
      randI = randomK.uniform(0, (*_learningAffectations)[randK].size());
      index = (*_learningAffectations)[randK][randI];
      (*_neuralNets)[_k]->forward((*_regData)[index][0]);
     for (uint v =0 ; v <  ((*_regData)[index][0]).getLength();v++){
       if((*_regData)[index][0][v]==0.0){
          blackTarget[v] = 1.0;
        }   
        else{
            blackTarget[v] = 0.0; 
        } 
      }
      backwardTiedWeights((*_neuralNets)[_k], blackTarget, _learningRate*0.1);
    }

    /* forward backward good sample */
    index = (*_learningAffectations)[_k][i];
    (*_neuralNets)[_k]->forward((*_regData)[index][0]);
    backwardTiedWeights((*_neuralNets)[_k], (*_regData).getTargetSample(index,0), _learningRate*10);
  }
}


/* Population trainer methods */
PopulationInverseTrainer::PopulationInverseTrainer(PBDNN& _population, RegressionDataset& _data, LearningParams& _params, RegressionDataset& _valid, Mask& _featureMask, Mask& _indexMask, ostream& _log) : SupervisedTrainer(_population, _data, _featureMask, _indexMask, _log), population(_population), params(_params), regData(_data), validationDataset(_valid), endurance(vector<uint>(population.getPopulation().size(),params.getDodges())){

}

void PopulationInverseTrainer::train() {
  uint i = params.getActualIteration();
  do {
    i++;
    log << "Iteration" << i << endl;
    trainOneIteration();
    params.setActualIteration(i);
    params.setLearningRate(params.getLearningRate() * params.getLearningRateDecrease());
    params.setErrorToFirst(params.getErrorToFirst() * params.getErrorToFirstIncrease());
    if (params.isSavedDuringProcess()) {
      ostringstream name;
      name << params.getSaveLocation() << "/populationIteration" << i << ".txt";
      ofstream outStream(name.str().c_str());
      outStream << this->population;
      outStream << this->params;
      outStream.close();
    }
    if (params.isValidatedDuringProcess() && (i+1)%params.getValidateEveryNIteration()==0) {
      log << "Validation "<< endl;
      validateIteration();
    }
  } while (i < params.getMaxIterations());
}

void PopulationInverseTrainer::trainOneIteration() {
  vector<uint> indexOrderSelection = defineIndexOrderSelection(data.getNumSequences());
  uint numberOfElementsToProcess = regData.getNumSequences()*params.getMaxTrainedPercentage();
  vector<NeuralNetworkPtr> neuralNets = population.getPopulation();
  AEMeasurer mae;
  DiversityMeasurer diversity = DiversityMeasurer(population, validationDataset, mae);
  uint maxTrained = regData.getNumSequences()*params.getMaxTrainedPercentage()/neuralNets.size();
  vector< vector<realv> > errors;
  vector<uint> histogramOfTrainees(neuralNets.size(),0);
  for (uint k = 0; k < neuralNets.size(); k++) {
    errors.push_back(vector<realv>(numberOfElementsToProcess,10e+9));
  }
  /* Use threaded forward process per network */
  vector<boost::thread * > threadsForward;
  for(uint k=0; k<neuralNets.size();k++){
    threadsForward.push_back(new boost::thread(threadForwardPerNetwork,&neuralNets, k, &regData, numberOfElementsToProcess, &indexOrderSelection, &errors));
  }
  for(uint k=0; k<neuralNets.size();k++){
    threadsForward[k]->join();
    delete threadsForward[k];
  }

  /* Find learning affectations, regenerate if endurance is 0*/
  realv maxError = params.getProximity()*((realv)neuralNets[0]->getInputLayer()->getNumUnits());
  vector<vector<uint> >learningAffectations = determineLearningAffectations(errors, indexOrderSelection, numberOfElementsToProcess, maxError);
  for(uint k=0; k<neuralNets.size();k++){
    log << "Network "<< k << " obtained " << learningAffectations[k].size() <<" samples" << endl;
    if(learningAffectations[k].size()==0){
      endurance[k] = endurance[k] - 1;
      if(endurance[k] == 0){
	endurance[k] = params.getDodges();
	population.regenerate(k);
      }
   } 
    else{
      endurance[k] = params.getDodges();
    }
  }
  /*Train according to learning affectations*/
  vector<boost::thread * > threadsForwardBackward;
  for(uint k=0; k<neuralNets.size();k++){
    threadsForwardBackward.push_back(new boost::thread(threadForwardBackwardPerNetwork, &neuralNets, k, &regData, &learningAffectations, params.getLearningRate(), maxTrained));
  }
  for(uint k=0; k<neuralNets.size();k++){
    threadsForwardBackward[k]->join();
    delete threadsForwardBackward[k];
  }
}

void PopulationInverseTrainer::validateIteration() {
  AEMeasurer mae;
  vector<NeuralNetworkPtr> neuralNets = population.getPopulation();
  DiversityMeasurer diversity = DiversityMeasurer(population, validationDataset, mae,params.getMaxTrainedPercentage());
  vector<int> validationAffectations = diversity.sampleRepartition();
  vector<realv> bestErrors = diversity.errorsOnBestSample();
  log << "network  | \t Global error \t|\t Best errors \t| \t timesSelected" << endl;
  for (uint i = 0; i < neuralNets.size(); i++) {
    RegressionMeasurer regMeasurer = RegressionMeasurer(*(neuralNets[i].get()), regData, mae, params.getMaxTrainedPercentage());
    regMeasurer.measurePerformance();
    log << i << " \t | \t " << regMeasurer.getTotalError() << "\t | \t" << bestErrors[i] << "\t | \t" << validationAffectations[i] << endl;
  }
  diversity.measurePerformance();
  log << "Diversity : " << endl << diversity.getDisagreementMatrix() << endl;
  log << "Diversity scalar: " << endl << diversity.getDisagreementScalar() << endl;
}

PopulationInverseTrainer::~PopulationInverseTrainer() {

}

vector<vector<uint > > PopulationInverseTrainer::determineLearningAffectations(vector<vector<realv> >& _errors, vector<uint>& _indexOrderSelection, uint _numberOfElementsToProcess, realv _maxError){
  vector<vector<uint > > learningAffectations = vector<vector<uint> >();
  for(uint k=0;k<_errors.size();k++){
    learningAffectations.push_back(vector<uint >());
  }
  uint bestNetwork = 0;
  realv minError = 10e+9;
  for(uint j=0;j<_numberOfElementsToProcess;j++){
    bestNetwork = 0;
    minError = 10e+9;
    for(uint k=0;k<_errors.size();k++){
      if(_errors[k][j]<minError){
	minError = _errors[k][j];
	bestNetwork = k;
      }
    }
    if(minError < _maxError){
      learningAffectations[bestNetwork].push_back(_indexOrderSelection[j]);
    }
  }
  return learningAffectations;
}

