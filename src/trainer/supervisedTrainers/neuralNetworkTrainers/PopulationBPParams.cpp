/*!
 * \file PopulationBPParams.cpp
 * Body of the PopulationBPParams class.
 * \author Luc Mioulet
 */

#include "PopulationBPParams.hpp"

using namespace cv;
using namespace std;

PopulationBPParams::PopulationBPParams(realv _learningRate, realv _learningRateDecrease, uint _maxIterations, uint _actualIteration, uint _maxTrained, realv _maxTrainedPercentage,
		realv _errorToFirst, realv _errorToFirstIncrease, bool _savedDuringProcess, string _saveLocation) :
		learningRate(_learningRate), learningRateDecrease(_learningRateDecrease), maxIterations(_maxIterations), maxTrained(_maxTrained), maxTrainedPercentage(
				_maxTrainedPercentage), errorToFirst(_errorToFirst), errorToFirstIncrease(_errorToFirstIncrease), savedDuringProcess(_savedDuringProcess), saveLocation(
				_saveLocation) {

}

realv PopulationBPParams::getLearningRate() const {
	return learningRate;
}

realv PopulationBPParams::getLearningRateDecrease() const {
	return learningRateDecrease;
}

uint PopulationBPParams::getMaxIterations() const {
	return maxIterations;
}

uint PopulationBPParams::getMaxTrained() const {
	return maxTrained;
}

realv PopulationBPParams::getErrorToFirst() const {
	return errorToFirst;
}

realv PopulationBPParams::getErrorToFirstIncrease() const {
	return errorToFirstIncrease;
}

void PopulationBPParams::setLearningRate(realv _learningRate) {
	learningRate = _learningRate;
}

void PopulationBPParams::setLearningRateDecrease(realv _learningRateDecrease) {
	learningRateDecrease = _learningRateDecrease;
}

void PopulationBPParams::setMaxIterations(uint _maxIterations) {
	maxIterations = _maxIterations;
}

void PopulationBPParams::setMaxTrained(uint _maxTrained) {
	maxTrained = _maxTrained;
}

void PopulationBPParams::setErrorToFirst(realv _errorToFirst) {
	if (_errorToFirst < 1.0) {
		errorToFirst = _errorToFirst;
	} else {
		errorToFirst = 1.0;
	}
}

void PopulationBPParams::setErrorToFirstIncrease(realv _errorToFirstIncrease) {
	errorToFirstIncrease = _errorToFirstIncrease;
}

realv PopulationBPParams::getMaxTrainedPercentage() const {
	return maxTrainedPercentage;
}

void PopulationBPParams::setMaxTrainedPercentage(realv maxTrainedPercentage) {
	if (maxTrainedPercentage <= 1.0 && maxTrainedPercentage >= 0.0) {
		this->maxTrainedPercentage = maxTrainedPercentage;
	}
}

bool PopulationBPParams::isSavedDuringProcess() const {
	return savedDuringProcess;
}

void PopulationBPParams::setSavedDuringProcess(bool _saveDuringProcess) {
	this->savedDuringProcess = _saveDuringProcess;
}

std::string PopulationBPParams::getSaveLocation() const {
	return saveLocation;
}

void PopulationBPParams::setSaveLocation(std::string _saveLocation) {
	this->saveLocation = _saveLocation;
}

uint PopulationBPParams::getActualIteration() const {
	return actualIteration;
}

void PopulationBPParams::setActualIteration(uint _actualIteration) {
	this->actualIteration = _actualIteration;
}

PopulationBPParams::~PopulationBPParams() {

}

ofstream& operator<<(ofstream& _ofs, const PopulationBPParams& _p) {
	_ofs << "< ";
	_ofs << " " << _p.getErrorToFirst();
	_ofs << " " << _p.getErrorToFirstIncrease();
	_ofs << " " << _p.getLearningRate();
	_ofs << " " << _p.getLearningRateDecrease();
	_ofs << " " << _p.getMaxIterations();
	_ofs << " " << _p.getActualIteration();
	_ofs << " " << _p.getMaxTrainedPercentage();
	_ofs << " " << _p.isSavedDuringProcess();
	_ofs << " " << _p.getSaveLocation();
	_ofs << " >";
	return _ofs;
}

ifstream& operator>>(ifstream& _ifs, PopulationBPParams& _p) {
	string stringValues;
	realv realValues;
	int intValues;
	bool boolValues;
	_ifs >> stringValues;
	_ifs >> realValues;
	_p.setErrorToFirst(realValues);
	_ifs >> realValues;
	_p.setErrorToFirstIncrease(realValues);
	_ifs >> realValues;
	_p.setLearningRate(realValues);
	_ifs >> realValues;
	_p.setLearningRateDecrease(realValues);
	_ifs >> intValues;
	_p.setMaxIterations(intValues);
	_ifs >> intValues;
	_p.setActualIteration(intValues);
	_ifs >> realValues;
	_p.setMaxTrainedPercentage(realValues);
	_ifs >> boolValues;
	_p.setSavedDuringProcess(boolValues);
	_ifs >> stringValues;
	_p.setSaveLocation(stringValues);
	_ifs >> stringValues;
	return _ifs;
}
