/*!
 * \file RTRL.cpp
 * Body of the RTRL class.
 * \author Luc Mioulet
 */

#include "RTRL.hpp"
#include <iostream>

using namespace cv;
using namespace std;

RTRL::RTRL(NeuralNetwork& _neuralNet, SupervisedDataset& _data, BackPropParams& _bpparams, Mask& _featureMask, Mask& _indexMask) :
		NeuralNetworkTrainer(_neuralNet, _data, _featureMask, _indexMask, _bpparams.getDoStochastic()), bpp(_bpparams) {
		vector<ConnectionPtr> connections = _neuralNet.getConnections();
		for(uint i=0;i<connections.size();i++){
			pWeights.push_back(Mat::zeros((connections[i].getWeights().size)));
		}
}

void RTRL::train() {
	uint i = 0;
	realv change = bpp.getMinChangeError() + 1.0;
	do {
		i++;
		trainOneIteration();
		if (i > 2) {
			change = abs(errorPerIteration[i - 2] - errorPerIteration[i - 1]);
		}
		bpp.setLearningRate(bpp.getLearningRate() * bpp.getLearningRateDecrease());
		cout << "Iteration : " << i << " ; Error : " << errorPerIteration[i - 1] << endl;
	} while (i < bpp.getMaxIterations()/* && change>bpp.getMinChangeError()*/&& errorPerIteration[i - 1] > bpp.getMinError());
}

void RTRL::trainOneIteration() {
	vector<uint> indexOrderSelection = defineIndexOrderSelection(data.getNumSequences());
	cout << indexOrderSelection[0] << endl;
	uint index = 0;
	uint sequenceIndex = 0;
	realv error = 0;
	vector<FeatureVector> sequence;
	for (uint i = 0; i < data.getNumSequences(); i++) {
		index = indexOrderSelection[i];
		sequence = data[index];
		resetSensitivity();
		for (uint j = 0; j < sequence.size(); j++) {
			if (neuralNet.isForward()) {
				sequenceIndex = j;
			} else {
				sequenceIndex = sequence.size() - j - 1;
			}
			neuralNet.forward(data[index][sequenceIndex]);
			backward(target,bpp.getLearningRate());
		}
		/*if(neuralNet.isForward()){
		 for(uint j=0;j<data[index].size() ; j++){
		 neuralNet.forward(data[index][j]);
		 FeatureVector target;
		 if(bpp.getTask()!=BP_AUTOENCODER){
		 target=trainData.getTargetSample(index,j);
		 }
		 else{
		 target=data[index][j];
		 }
		 backward(target, bpp.getLearningRate());
		 error+=measureSampleError(neuralNet.getOutputSignal(),target);
		 }
		 }*/
	}
	cout << error / data.getNumSequences() << endl;
	errorPerIteration.push_back(error / data.getNumSequences());
}

realv RTRL::measureSampleError(FeatureVector networkOutput, FeatureVector target) {
	realv error = 0;
	if (bpp.getTask() == BP_CLASSIFICATION) {
		ClassificationErrorMeasurer ce;
		error += ce.totalError(networkOutput, target);
	} else {
		SEMeasurer mse;
		error += mse.totalError(networkOutput, target);
	}
	return error;
}

void RTRL::backward(FeatureVector _target, realv _learningRate) {
	vector<ConnectionPtr> connections = neuralNet.getConnections();
	vector<LayerPtr> layers = neuralNet.getHiddenLayers();
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
		updateConnection(connections[i], deltas[connections.size() - i - 1], _learningRate);
	}
}

ErrorVector RTRL::calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta) {
	ErrorVector delta = ErrorVector(_layer->getNumUnits());
	for (uint i = 0; i < delta.getLength(); i++) {
		delta[i] = _derivatives[i] * _layer->errorWeighting(_previousLayerDelta, _layer->getOutputConnection()->getWeightsFromNeuron(i));
	}
	return delta;
}

ErrorVector RTRL::calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives) {
	ErrorVector delta = ErrorVector(_layer->getNumUnits());
	for (uint i = 0; i < _target.getLength(); i++) {
		delta[i] = _derivatives[i] * (_target[i] - _layer->getOutputSignal()[i]); // error calculation if output layer
	}
	return delta;
}

void RTRL::updateConnection(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate) {
	Mat weights = _connection->getWeights();
	for (int i = 0; i < weights.rows; i++) {
		for (int j = 0; j < weights.cols; j++) {
			weights.at<realv>(i, j) = weights.at<realv>(i, j) + _learningRate * _deltas[i] * _connection->getInputLayer()->getOutputSignal()[j];
		}
	}
	_connection->setWeights(weights);
}

void RTRL::resetPWeights(){
	vector<ConnectionPtr> connections = neuralNet.getConnections();
	for(uint i=0;i<connections.size();i++){
			pWeights.push_back(Mat::zeros((connections[i]->getWeights().size)));
	}
}

RTRL::~RTRL() {

}
