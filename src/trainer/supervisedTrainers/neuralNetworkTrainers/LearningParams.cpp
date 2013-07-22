/*!
 * \file LearningParams.cpp
 * Body of the LearningParams class.
 * \author Luc Mioulet
 */

#include "LearningParams.hpp"

using namespace cv;
using namespace std;

LearningParams::LearningParams() :learningRate(0.001), learningRateDecrease(0.95), maxIterations(10), actualIteration(0), minError(0.0),minChangeError(0.05),maxTrained(5), maxTrainedPercentage(0.25), errorToFirst(0.5), errorToFirstIncrease(1.1), savedDuringProcess(false),
    saveLocation("."), validatedDuringProcess(true), validateEveryNIteration(5), task(BP_REGRESSION),stochastic(true),dodges(2), proximity(0.30) , noise(0.0), interFrameSpace(0) {

}

realv LearningParams::getMinError(){
  return minError;
}

realv LearningParams::getMinChangeError(){
  return minChangeError;
}

realv LearningParams::getLearningRate() const {
	return learningRate;
}

realv LearningParams::getLearningRateDecrease() const {
	return learningRateDecrease;
}

uint LearningParams::getMaxIterations() const {
	return maxIterations;
}

uint LearningParams::getMaxTrained() const {
	return maxTrained;
}

realv LearningParams::getErrorToFirst() const {
	return errorToFirst;
}

realv LearningParams::getErrorToFirstIncrease() const {
	return errorToFirstIncrease;
}

uint LearningParams::getDodges() const {
	return dodges;
}

uint LearningParams::getActualIteration() const {
	return actualIteration;
}

realv LearningParams::getProximity() const {
  return proximity;
}

realv LearningParams::getMaxTrainedPercentage() const {
	return maxTrainedPercentage;
}

realv LearningParams::getNoise() const {
	return noise;
}

std::string LearningParams::getSaveLocation() const {
	return saveLocation;
}

void LearningParams::setLearningRate(realv _learningRate) {
	learningRate = _learningRate;
}

void LearningParams::setLearningRateDecrease(realv _learningRateDecrease) {
	learningRateDecrease = _learningRateDecrease;
}

void LearningParams::setMaxIterations(uint _maxIterations) {
	maxIterations = _maxIterations;
}

void LearningParams::setMaxTrained(uint _maxTrained) {
	maxTrained = _maxTrained;
}

void LearningParams::setErrorToFirst(realv _errorToFirst) {
	if (_errorToFirst < 1.0) {
		errorToFirst = _errorToFirst;
	} else {
		errorToFirst = 1.0;
	}
}

void LearningParams::setErrorToFirstIncrease(realv _errorToFirstIncrease) {
	errorToFirstIncrease = _errorToFirstIncrease;
}

void LearningParams::setMaxTrainedPercentage(realv maxTrainedPercentage) {
	if (maxTrainedPercentage <= 1.0 && maxTrainedPercentage >= 0.0) {
		this->maxTrainedPercentage = maxTrainedPercentage;
	}
}

bool LearningParams::isSavedDuringProcess() const {
	return savedDuringProcess;
}

void LearningParams::setSavedDuringProcess(bool _saveDuringProcess) {
	this->savedDuringProcess = _saveDuringProcess;
}


void LearningParams::setSaveLocation(std::string _saveLocation) {
	this->saveLocation = _saveLocation;
}

void LearningParams::setDodges(uint _dodges) {
	this->dodges = _dodges;
}


void LearningParams::setActualIteration(uint _actualIteration) {
	this->actualIteration = _actualIteration;
}

bool LearningParams::isValidatedDuringProcess() const {
	return validatedDuringProcess;
}

void LearningParams::setValidatedDuringProcess(bool _validatedDuringProcess) {
	this->validatedDuringProcess = _validatedDuringProcess;
}

void LearningParams::setMinChangeError(realv _minChangeError) {
	this->minChangeError = _minChangeError;
}

void LearningParams::setMinError(realv _minError) {
	this->minError = _minError;
}

bool LearningParams::isStochastic() const {
	return stochastic;
}

int LearningParams::getTask() const {
	return task;
}

void LearningParams::setTask(int _task) {
	this->task = task;
}

void LearningParams::setStochastic(bool _stochastic) {
	this->stochastic = _stochastic;
}

void LearningParams::setProximity(realv _proximity){
  this->proximity = _proximity;
}

void LearningParams::setNoise(realv _noise){
  this->noise = _noise;
}

LearningParams::~LearningParams() {

}

ofstream& operator<<(ofstream& _ofs, const LearningParams& _p) {
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
	_ofs << " " << _p.getNoise();
	_ofs << " >";
	return _ofs;
}

int LearningParams::getValidateEveryNIteration() const {
	return validateEveryNIteration;
}

void LearningParams::setValidateEveryNIteration(int validateEveryNIteration) {
	this->validateEveryNIteration = validateEveryNIteration;
}

ifstream& operator>>(ifstream& _ifs, LearningParams& _p) {
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
	_ifs >> realValues;
	_p.setNoise(realValues);
	_ifs >> stringValues;

	return _ifs;
}
