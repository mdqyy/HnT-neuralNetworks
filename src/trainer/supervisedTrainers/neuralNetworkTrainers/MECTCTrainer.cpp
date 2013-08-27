/*!
 * \file LayerMECTC.cpp
 * Body of the LayerMECTC class.
 * \author Luc Mioulet
 */

#include "MECTCTrainer.hpp"

using namespace std;
using namespace cv;

MECTCTrainer::MECTCTrainer(MixedEnsembles& _machine, ImageDataset& _trainingData, SequenceClassDataset& _trainingSequences, ImageDataset& _validationData, SequenceClassDataset& _validationSequences, LearningParams _params,ostream& _log) : machine(_machine), trainingData(_trainingData), trainingSequences(_trainingSequences),validationData(_validationData), validationSequences(_validationSequences),params(_params), log(_log) {
  ctcLayer = (LayerCTC*)machine.getOutputNetwork()->getOutputLayer();
}

void MECTCTrainer::train() {
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
			name << params.getSaveLocation() << "/ctcIteration" << i << ".txt";
			ofstream outStream(name.str().c_str());
			outStream << this->ctcLayer;
			outStream << this->params;
		}
		if (params.isValidatedDuringProcess()) {
		 log << "Validation "<< endl;
		 validateIteration();
		 }
	} while (i < params.getMaxIterations());
}

vector<uint> MECTCTrainer::defineIndexOrderSelection(uint _numSequences) {
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

void MECTCTrainer::trainOneIteration() {
	vector<uint> indexOrderSelection = defineIndexOrderSelection(trainingData.getNumberOfImages());
	 uint numberOfElementsToProcess = trainingData.getNumberOfImages()* params.getMaxTrainedPercentage();
	uint index = 0;
	for (uint i = 0; i < numberOfElementsToProcess; i++) {
		index = indexOrderSelection[i];
		log << i << " "<<trainingData.getImage(index);
		Mat image = trainingData.getMatrix(index);
		vector<int> targetSignal = trainingSequences.getSequenceClassesIndex(index);
		trainOneSample(image, targetSignal);
		log << " trained image" << endl;
	}
}

void MECTCTrainer::trainOneSample(Mat _image, vector<int> _targetSignal) {
	uint requiredTime = calculateRequiredTime(_targetSignal);
	if (requiredTime > _image.cols) {
		throw length_error("MECTCTrainer : Required time is superior to input signal size. Learning from this sample is impossible.");
	}
	vector<FeatureVector> outputSignals = vector<FeatureVector>();
	for(uint i=0;i<_image.cols;i++){
	  machine.forwardOnPixel(_image,i);
	  outputSignals.push_back(machine.getOutput());
	}
	vector<ValueVector> forwardVariables = processForwardVariables(outputSignals, _targetSignal);
	vector<ValueVector> backwardVariables = processBackwardVariables(outputSignals, _targetSignal);
	vector<ErrorVector> derivatives = processDerivatives(_targetSignal, outputSignals, forwardVariables, backwardVariables);
	backwardSequence(derivatives);
}

void MECTCTrainer::backwardSequence(std::vector<ErrorVector> _derivatives/*worst name ever, actually is delta*/) {
    NeuralNetworkPtr outputNet = machine.getOutputNetwork();
    vector<ConnectionPtr> connections = outputNet->getConnections();
    vector<LayerPtr> layers = outputNet->getHiddenLayers();
    vector<ErrorVector> deltas = vector<ErrorVector>();/* pushed in inversed order so be careful */
	for (uint t = 0; t < _derivatives.size(); t++) {
		updateConnection(ctcLayer->getInputConnection(), _derivatives[t]); /* update first on CTC connections */
	    for (uint i = layers.size() - 2; i > 0; i--) { /* Then on the underlying connections */
	      ValueVector derivatives = layers[i]->getDerivatives();
	      if(i==layers.size() - 2){
	        deltas.push_back(calculateDeltas(layers[i],derivatives, _derivatives[t]));
	      }
	      else{
	      deltas.push_back(calculateDeltas(layers[i],derivatives, deltas[deltas.size() - 1]));
	      }
	    }
	}
}

ErrorVector MECTCTrainer::calculateDeltas(LayerPtr _layer, ValueVector _derivatives, ErrorVector _previousLayerDelta) {
  ErrorVector delta = ErrorVector(_layer->getNumUnits());
  for (uint i = 0; i < delta.getLength(); i++) {
    delta[i] = _derivatives[i] * _layer->errorWeighting(_previousLayerDelta, _layer->getOutputConnection()->getWeightsFromNeuron(i));
  }
  return delta;
}

void MECTCTrainer::updateConnection(Connection* _connection, ErrorVector _deltas) {
	Mat weights = _connection->getWeights();
	for (int i = 0; i < weights.rows; i++) {
		for (int j = 0; j < weights.cols; j++) {
			weights.at<realv>(i, j) = weights.at<realv>(i, j) + _deltas[i] * _connection->getInputLayer()->getOutputSignal()[j];
		}
	}
	_connection->setWeights(weights);
}

uint MECTCTrainer::calculateRequiredTime(vector<int> _targetSignal) const {
	uint requiredTime = _targetSignal.size();
	int previousLabel = -1;
	for (uint i = 0; i < _targetSignal.size(); i++) {
		if (_targetSignal[i] == previousLabel) {
			requiredTime++;
		}
		previousLabel = _targetSignal[i];
	}
	return requiredTime;
}

vector<ValueVector> MECTCTrainer::processForwardVariables(vector<FeatureVector> _outputSignals, vector<int> _targetSequence) {
	uint blankIndex = _outputSignals[0].getLength() - 1; //position of the blank output
	uint requiredSegments = 2 * _targetSequence.size() + 1; //required segments to model a sequence
	int previousLabel = -1; //previous label value
	vector<ValueVector> forwardVariables = vector<ValueVector>(_outputSignals.size(), ValueVector(requiredSegments));
	normalizeC = vector<realv>(_outputSignals.size(), 0.0);
	forwardVariables[0][0] = _outputSignals[0][blankIndex];
	forwardVariables[0][1] = _outputSignals[0][_targetSequence[0]];
	for (uint t = 1; t < _outputSignals.size(); t++) {
		uint minLabels = determineMinLabel(t, _outputSignals.size(), requiredSegments, _targetSequence.size());
		uint maxLabels = determineMaxLabel(t, _targetSequence.size());
		for (uint s = minLabels; s < maxLabels; s++) {
			if (s % 2 == 0) { /* is even : blank*/
				if (s > 0) {
					forwardVariables[t][s] = _outputSignals[t][blankIndex] * (forwardVariables[t - 1][s - 1] + forwardVariables[t - 1][s]);
				} else {
					forwardVariables[t][s] = _outputSignals[t][blankIndex] * (forwardVariables[t - 1][s]);
				}
			} else { /*is odd : label */
				int label = _targetSequence[s / 2];
				int problemLength = _outputSignals.size();
				if (s >= 2 && label != previousLabel) {
				  int probLen = _outputSignals[t].getLength();
					forwardVariables[t][s] = _outputSignals[t][label]
							* (forwardVariables[t - 1][s - 2] + forwardVariables[t - 1][s - 1] + forwardVariables[t - 1][s]);
				} else {
					forwardVariables[t][s] = _outputSignals[t][label] * (forwardVariables[t - 1][s - 1] + forwardVariables[t - 1][s]);
				}
				previousLabel = label;
			}
			normalizeC[t] += forwardVariables[t][s];
		}
		for (uint s = minLabels; s < maxLabels; s++) {
			forwardVariables[t][s] /= normalizeC[t];
		}
	}
	return forwardVariables;
}

vector<ValueVector> MECTCTrainer::processBackwardVariables(vector<FeatureVector> _outputSignals, vector<int> _targetSequence) {
	uint blankIndex = _outputSignals[0].getLength() - 1;
	uint requiredSegments = 2 * _targetSequence.size() + 1;
	int previousLabel = -1;
	vector<ValueVector> backwardVariables = vector<ValueVector>(_outputSignals.size(), ValueVector(requiredSegments));
	normalizeD = vector<realv>(_outputSignals.size(), 0.0);
	backwardVariables[_outputSignals.size() - 1][requiredSegments - 1] = 1;
	backwardVariables[_outputSignals.size() - 1][requiredSegments - 1] = 1;
	for (uint t = _outputSignals.size() - 2; t < 0; t--) {
		uint minLabels = determineMinLabel(t, _outputSignals.size(), requiredSegments, _targetSequence.size());
		uint maxLabels = determineMaxLabel(t, _targetSequence.size());
		for (uint s = minLabels; s < maxLabels; s++) {
			if (s % 2 == 0) { /* is even : blank*/
				if (s < _targetSequence.size() * 2 - 1) {
					backwardVariables[t][s] = _outputSignals[t][blankIndex] * (backwardVariables[t + 1][s + 1] + backwardVariables[t + 1][s]);
				} else {
					backwardVariables[t][s] = _outputSignals[t][blankIndex] * (backwardVariables[t + 1][s]);
				}
			} else { /*is odd : label */
				int label = _targetSequence[s / 2];
				if (s >= 2 && label != previousLabel) {
					backwardVariables[t][s] = _outputSignals[t][label]
							* (backwardVariables[t + 1][s + 2] + backwardVariables[t + 1][s + 1] + backwardVariables[t + 1][s]);
				} else {
					backwardVariables[t][s] = _outputSignals[t][label]
							* (backwardVariables[t + 1][s + 2] + backwardVariables[t + 1][s + 1] + backwardVariables[t + 1][s]);
				}
				previousLabel = label;
			}
			normalizeD[t] += backwardVariables[t][s];
		}
		for (uint s = minLabels; s < maxLabels; s++) {
			backwardVariables[t][s] /= normalizeD[t];
		}
	}
	return backwardVariables;
}

vector<ErrorVector> MECTCTrainer::processDerivatives(vector<int> _targetSignal, vector<FeatureVector> _outputSignals, vector<ValueVector> _forwardVariables,
		vector<ValueVector> _backwardVariables) {
	uint blankIndex = _outputSignals[0].getLength() - 1;
	vector<ErrorVector> derivatives = vector<ErrorVector>(_outputSignals.size(), ErrorVector(_outputSignals[0].getLength()));
	vector<realv> normalizeQ = processQ();
	vector<int> uniqueTargetLabels = findUniqueElements(_targetSignal);
	for (uint t = 0; t < _outputSignals.size(); t++) {
		for (uint l = 0; l < _targetSignal.size(); l++) {
		    int tsl = _targetSignal[l];
			derivatives[t][blankIndex] += _forwardVariables[t][2 * l] * _backwardVariables[t][2 * l];
			derivatives[t][_targetSignal[l]] += _forwardVariables[t][2 * l + 1] * _backwardVariables[t][2 * l + 1];
		}
		for (uint l = 0; l < uniqueTargetLabels.size(); l++) {
			derivatives[t][uniqueTargetLabels[l]] = _outputSignals[t][uniqueTargetLabels[l]]
					- (normalizeQ[t]) / (_outputSignals[t][uniqueTargetLabels[l]]) * derivatives[t][uniqueTargetLabels[l]]; /*! \todo perhaps change this calculation */
		}
		derivatives[t][blankIndex] = _outputSignals[t][blankIndex] - (normalizeQ[t]) / (_outputSignals[t][blankIndex]) * derivatives[t][blankIndex]; /*! \todo perhaps change this calculation */
	}
	return derivatives;
}

std::vector<realv> MECTCTrainer::processQ() const {
	vector<realv> normalizeQ = vector<realv>(normalizeC.size(), 0.0);
	realv piDivisionDC = 1;
	for (uint t = 0; t < normalizeC.size(); t++) {
		piDivisionDC = 1;
		for (uint f = t + 1; f < normalizeC.size(); f++) {
			piDivisionDC *= normalizeC[f] / normalizeD[f];
		}
		normalizeQ[t] = normalizeD[t] * piDivisionDC;
	}
	return normalizeQ;
}

vector<int> MECTCTrainer::findUniqueElements(vector<int> _targetSignal) {
	vector<int> uniques = vector<int>(_targetSignal);
	sort(uniques.begin(), uniques.end());
	vector<int>::iterator it;
	it = unique(uniques.begin(), uniques.end());
	uniques.resize(it - uniques.begin());
	return uniques;
}

uint MECTCTrainer::determineMaxLabel(uint _t, uint _targetSequenceSize) {
	uint maxLabels = (_t + 1) * 2;
	if (maxLabels > _targetSequenceSize) {
		maxLabels = _targetSequenceSize;
	}
	return maxLabels;
}

uint MECTCTrainer::determineMinLabel(uint _t, uint _outputSignalsSize, uint _requiredSegments, uint _targetSequenceSize) {
	uint minLabels = 0;
	if (_outputSignalsSize - _t - 1 < _targetSequenceSize) {
		minLabels = _requiredSegments - (_outputSignalsSize - _t) * 2;
	}
	return minLabels;
}

void MECTCTrainer::validateIteration() {
//	CERMEasurer mae;

	//log << "Target \t|\t Output\t| "<< endl;
	for(uint i = 0; i < validationData.getNumberOfImages();i++){
	    Mat image = validationData.getMatrix(i);
	    vector<FeatureVector> outputSignals;
	    uint werGood = 0;
	    uint werBad = 0;
	    for(uint j=0;j<image.cols;j++){
	      machine.forwardOnPixel(image,j);
	      outputSignals.push_back(machine.getOutput());
	    }
		ctcLayer->setOutputSignals(outputSignals);
		vector<string> target = validationSequences.getSequenceClasses(i);
		string truth = "";
		log << "truth :" ;
		for(uint j = 0; j<target.size();j++){
			log << target[j];
			truth.append(target[j]);
		}
		log <<"  :" <<ctcLayer->outputWord()<< endl;
		if(truth.compare(ctcLayer->outputWord())){
		  werGood++;
		}
		else{
		  werBad++;
		}
		log << "wer good: " << werGood << endl;
		log << "wer bad : " << werBad << endl;
	}
}

MECTCTrainer::~MECTCTrainer() {

}

