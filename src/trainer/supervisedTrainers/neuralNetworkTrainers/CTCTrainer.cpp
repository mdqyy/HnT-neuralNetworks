/*!
 * \file LayerCTC.cpp
 * Body of the LayerCTC class.
 * \author Luc Mioulet
 */

#include "CTCTrainer.hpp"

using namespace std;
using namespace cv;

CTCTrainer::CTCTrainer(LayerCTC& _ctcLayer, ClassificationDataset& _data, Mask& _featureMask, Mask& _indexMask) : SupervisedTrainer(_ctcLayer,_data,_featureMask,_indexMask), ctcLayer(_ctcLayer), classifactionData(_data) {

}

void CTCTrainer::train(){

}

void CTCTrainer::trainOneIteration(){
	vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
	uint index=0;
	for(uint i=0;i<classifactionData.numSequences();i++){
		index = indexOrderSelection[i];
		vector<FeatureVector> inputSignal = classifactionData[index];
		vector<int> targetSignal = classifactionData.getSequenceClassesIndex(index);
		this->trainOneSample(inputSignal,targetSignal);
	}
}

void CTCTrainer::trainOneSample(vector<FeatureVector> _inputSignal, vector<int> _targetSignal){
	uint requiredTime = this->calculateRequiredTime(_targetSignal);
	if(requiredTime > _inputSignal.size()){
		throw length_error("CTCTrainer : Required time is superior to input signal size. Learning from this sample is impossible.");
	}
	ctcLayer.forwardSequence(_inputSignal);
	vector<FeatureVector> outputSignals = ctcLayer.getOutputSignals();
	vector<ValueVector> forwardVariables = this->processForwardVariables(outputSignals,_targetSignal);
	vector<ValueVector> backwardVariables = this->processBackwardVariables(outputSignals,_targetSignal);
	vector<ErrorVector> derivatives = this->processDerivatives(_targetSignal, outputSignals,forwardVariables,backwardVariables);
	this->backwardSequence(derivatives);
}

void CTCTrainer::backwardSequence(std::vector<ErrorVector> _derivatives){
	for(uint t=0;t<_derivatives.size(); t++){
		this->updateConnection(this->ctcLayer.inputConnection,_derivatives[t]);
	}
}

void CTCTrainer::updateConnection(Connection* _connection, ErrorVector _deltas){
  Mat weights = _connection->getWeights();
  for(int i=0;i<weights.rows;i++){
    for(int j=0;j<weights.cols;j++){
      weights.at<realv>(i,j)=weights.at<realv>(i,j)+_deltas[i]*_connection->getInputLayer()->getOutputSignal()[j];
    }
  }
  _connection->setWeights(weights);
}

uint CTCTrainer::calculateRequiredTime(vector<int> _targetSignal) const{
	uint requiredTime = _targetSignal.size();
	int previousLabel = -1;
	for(uint i=0;i<_targetSignal.size();i++){
		if(_targetSignal[i] == previousLabel){
			requiredTime ++;
		}
		previousLabel = _targetSignal[i];
	}
	return requiredTime;
}

vector<ValueVector> CTCTrainer::processForwardVariables(vector<FeatureVector> _outputSignals, vector<int> _targetSequence){
	uint blankIndex = _outputSignals[0].getLength()-1;
	uint requiredSegments = 2*_targetSequence.size()+1;
	int previousLabel = -1;
	vector<ValueVector> forwardVariables = vector<ValueVector>(ValueVector(requiredSegments),_outputSignals.size());
	normalizeC = vector<realv>(_outputSignals,0.0);
	forwardVariables[0][0] = _outputSignals[0][blankIndex];
	forwardVariables[0][1] = _outputSignals[0][_targetSequence[0]];
	for(uint t=1;t<_outputSignals.size();t++){
		uint minLabels = this->determineMinLabel(t,_outputSignals.size(),requiredSegments,_targetSequence.size());
		uint maxLabels = this->determineMaxLabel(t,_targetSequence.size());
		for(uint s=minLabels;s<maxLabels;s++){
			if(s%2){ /* is even : blank*/
				if(s>0){
					forwardVariables[t][s]= _outputSignals[blankIndex]*(forwardVariables[t-1][s-1]+forwardVariables[t-1][s]);
				}
				else{
					forwardVariables[t][s]= _outputSignals[blankIndex]*(forwardVariables[t-1][s]);
				}
			}
			else{ /*is odd : label */
				int label = _targetSequence[s/2 -1];
				if(s>=2 && label!=previousLabel){
					forwardVariables[t][s]= _outputSignals[blankIndex]*(forwardVariables[t-1][s-2]+forwardVariables[t-1][s-1]+forwardVariables[t-1][s]);
				}
				else{
					forwardVariables[t][s]= _outputSignals[blankIndex]*(forwardVariables[t-1][s-1]+forwardVariables[t-1][s]);
				}
				previousLabel = label;
			}
			normalizeC[t]+=forwardVariables[t][s];
		}
		for(uint s=minLabels;s<maxLabels;s++){
			forwardVariables[t][s]/=normalizeC[t];
		}
	}
	return forwardVariables;
}

vector<ValueVector> CTCTrainer::processBackwardVariables(vector<FeatureVector> _outputSignals, vector<int> _targetSequence){
	uint blankIndex = _outputSignals[0].getLength()-1;
	uint requiredSegments = 2*_targetSequence.size()+1;
	int previousLabel = -1;
	vector<ValueVector> backwardVariables = vector(ValueVector(requiredSegments),_outputSignals.size());
	normalizeD = vector<realv>(_outputSignals,0.0);
	backwardVariables[_outputSignals.size()-1][requiredSegments-1] = 1;
	backwardVariables[_outputSignals.size()-1][requiredSegments-1] = 1;
	for(uint t=_outputSignals.size()-2;t<0;t--){
		uint minLabels = this->determineMinLabel(t,_outputSignals.size(),requiredSegments,_targetSequence.size());
		uint maxLabels = this->determineMaxLabel(t,_targetSequence.size());
		for(uint s=minLabels;s<maxLabels;s++){
			if(s%2){ /* is even : blank*/
				if(s>0){
					backwardVariables[t][s]= _outputSignals[blankIndex]*(backwardVariables[t+1][s+1]+backwardVariables[t+1][s]);
				}
				else{
					backwardVariables[t][s]= _outputSignals[blankIndex]*(backwardVariables[t+1][s]);
				}
			}
			else{ /*is odd : label */
				int label = _targetSequence[s/2 -1];
				if(s>=2){
					backwardVariables[t][s]= _outputSignals[blankIndex]*(backwardVariables[t+1][s+2]+backwardVariables[t+1][s+1]+backwardVariables[t+1][s]);
				}
				else{
					backwardVariables[t][s]= _outputSignals[blankIndex]*(backwardVariables[t+1][s+2]+backwardVariables[t+1][s+1]+backwardVariables[t+1][s]);
				}
				previousLabel=label;
			}
			normalizeD[t]+=backwardVariables[t][s];
		}
		for(uint s=minLabels;s<maxLabels;s++){
			backwardVariables[t][s]/=normalizeD[t];
		}
	}
	return backwardVariables;
}

vector<ErrorVector> CTCTrainer::processDerivatives(vector<int> _targetSignal,vector<FeatureVector> _outputSignals,vector<ValueVector> _forwardVariables,vector<ValueVector> _backwardVariables){
	vector<ErrorVector> derivatives = vector<ErrorVector>(ErrorVector(_outputSignals[0].getLength()),_outputSignals.size());
	vector<FeatureVector> normalizeQ = this->processQ();
	vector<int> uniqueTargetLabels = findUniqueElements(_targetSignal);
	for(uint t=0; t<_outputSignals.size();t++){
		for(uint l=0;l<_targetSignal.size();l++){
			derivatives[t][_targetSignal[l]] += _forwardVariables[t][2*l+1]*_forwardVariables[t][2*l+1];
		}
		for(uint l=0;l<uniqueTargetLabels.size();l++){
			derivatives[t][uniqueTargetLabels[l]] = _outputSignals[t][l] - normalizeQ[t]/_outputSignals[t][l]*derivatives[t][uniqueTargetLabels[l]]; /*! \todo perhaps change this calculation */
		}
	}
	return derivatives;
}

std::vector<realv> CTCTrainer::processQ() const{
	vector<realv> normalizeQ;
	realv piDivisionDC=1;
	for(uint t=0;t<normalizeC.size();t++){
		piDivisionDC=1;
		for(uint f=t+1;f<normalizeC.size();f++){
			piDivisionDC*=normalizeC[f]/normalizeD[f];
		}
		normalizeQ[t]=normalizeD[t] * piDivisionDC;
	}
	return normalizeQ;
}

vector<int> CTCTrainer::findUniqueElements(vector<int> _targetSignal){
	vector<int> uniques = vector<int>(_targetSignal);
	sort(uniques.begin(),uniques.end());
	vector<int>::iterator it;
	it = unique (uniques.begin(), uniques.end());
	uniques.resize( it - uniques.begin() );
	return uniques;
}

uint CTCTrainer::determineMaxLabel(uint _t, uint _targetSequenceSize){
	uint maxLabels = (_t+1)*2;
	if(maxLabels > _targetSequenceSize){
		maxLabels = _targetSequenceSize;
	}
	return maxLabels;
}

uint CTCTrainer::determineMinLabel(uint _t, uint _outputSignalsSize, uint _requiredSegments, uint _targetSequenceSize){
	uint minLabels = 0;
	if(_outputSignalsSize-_t-1<_targetSequenceSize){
		minLabels=_requiredSegments-(_outputSignalsSize-_t)*2;
	}
	return minLabels;
}

CTCTrainer::~CTCTrainer() {

}

