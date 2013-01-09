/*!
 * \file LayerCTC.cpp
 * Body of the LayerCTC class.
 * \author Luc Mioulet
 */

#include "LayerCTC.hpp"
#include <math.h>

using namespace std;
using namespace cv;

LayerCTC::LayerCTC() :
		Layer() {

}

LayerCTC::LayerCTC(uint _numUnits, string _name) :
		Layer(_numUnits, _name, false) {

}

LayerCTC::LayerCTC(const LayerCTC& _clsm) :
		Layer(_clsm) {

}

LayerCTC* LayerCTC::clone() const {
	return new LayerCTC(*this);
}

int LayerCTC::getLayerType() const {
	return LAYER_CTC;
}

map<int, string> LayerCTC::getClassLabelMap() {
	return classLabels;
}

map<string, int> LayerCTC::getClassLabelIndexMap() {
	return classLabelIndex;
}

void LayerCTC::forward() {
	outputSignal.reset(0.0);
	/* Accumulate neuron sum */
	FeatureVector layerInputSignal = getInputConnection()->getInputLayer()->getOutputSignal();
	forward(layerInputSignal);
	if (getOutputConnection() > 0) {
		getOutputConnection()->forward();
	}
}

void LayerCTC::forward(FeatureVector _signal) {
	networkInputSignal = _signal;
	realv sum = 0;
	for (uint i = 0; i < numUnits; i++) {
		outputSignal[i] = signalWeighting(createInputSignal(), getInputConnection()->getWeightsToNeuron(i));
		sum += outputSignal[i];
	}
	for (uint i = 0; i < numUnits; i++) {
		outputSignal[i] = outputSignal[i] / sum;
	}
	outputSignal[numUnits] = 1.0;
}

vector<int> LayerCTC::processResultSequence() {
	vector<int> result;
	for (uint i = 0; i < outputSignals.size(); i++) {
		realv maxOutput = 0;
		int bestNeuron = 0;
		for(uint j = 0; j<outputSignals[i].getLength();j++){
			if(outputSignals[i][j] > maxOutput){
				maxOutput = outputSignals[i][j];
				bestNeuron = j;
			}
		}
		result.push_back(bestNeuron);
	}
	return result;
}

vector<int> LayerCTC::processCleanedResultSequence() {
	vector<int> result;
	vector<int> temp = processResultSequence();
	if(temp[0]!=this->getNumUnits()-1 ){
		result.push_back(temp[0]);
	}
	for(int i=1;i<temp.size();i++){
		if(temp[i]!=temp[i-1]){
			if(temp[i]!=this->getNumUnits()-1 ){
					result.push_back(temp[i]);
			}
		}
	}
	return result;
}

string LayerCTC::outputWord(){
	string word;
	ostringstream stream;
	vector<int> temp = processCleanedResultSequence();
	for(int i = 0 ; i < temp.size(); i++){
		map<int,string>::iterator it = classLabels.find(temp[i]);
		stream << classLabels[temp[i]];
	}
	word = stream.str();
	return word;
}

FeatureVector LayerCTC::createInputSignal() {
	FeatureVector inSig = FeatureVector(inputConnection->getWeights().cols);
	for (uint i = 0; i < networkInputSignal.getLength(); i++) {
		inSig[i] = networkInputSignal[i];
	}
	inSig[inputConnection->getWeights().cols - 1] = 0;
	return inSig;
}

ValueVector LayerCTC::getDerivatives() const {
	ValueVector deriv = ValueVector(numUnits + 1);
	for (uint i = 0; i < deriv.getLength(); i++) {
		deriv[i] = 1.0;
	}
	return deriv;
}

vector<ValueVector> LayerCTC::getDerivatives(vector<FeatureVector> _forwardVariables, vector<FeatureVector> _backwardVariables) const {
	vector<ValueVector> derivatives = vector<ValueVector>();
	return derivatives;
}

LayerCTC::~LayerCTC() {

}

void LayerCTC::print(ostream& _os) const {
	_os << "CTC neuron layer : " << endl;
	if (isRecurrent()) {
		_os << "\t -A recurrent layer." << endl;
	}
	_os << "\t -Name :" << getName() << endl;
	_os << "\t -Units : " << getNumUnits() << endl;
}

ofstream& operator<<(ofstream& _ofs, const LayerCTC& _l) {
	_ofs << " < " << _l.getName() << " " << _l.getNumUnits() << " " << _l.isRecurrent();
	_ofs << " > " << endl;
	return _ofs;
}

void LayerCTC::setClassLabelIndex(std::map<std::string, int> classLabelIndex) {
	this->classLabelIndex = classLabelIndex;
}

void LayerCTC::setClassLabels(std::map<int, std::string> classLabels) {
	this->classLabels = classLabels;
}

ifstream& operator>>(ifstream& _ifs, LayerCTC& _l) {
	int nUnits;
	bool boolRecurrent;
	ValueVector meanV, stdV;
	/*  Connection recCo; */
	string name, temp;
	_ifs >> temp;
	_ifs >> name;
	_ifs >> nUnits;
	_ifs >> boolRecurrent;
	_ifs >> temp;
	_l.setName(name);
	_l.setNumUnits(nUnits);
	_l.setRecurent(boolRecurrent);
	return _ifs;
}
