/*!
 * \file RegressionMeasurer.cpp
 * Body of the RegressionMeasurer class.
 * \author Luc Mioulet
 */

#include "RegressionMeasurer.hpp"


RegressionMeasurer::RegressionMeasurer(NeuralNetwork& _machine, RegressionDataset& _data, ErrorMeasurer& _em) : machine(_machine), data(_data), errorMeasurer(_em), totalError(0.0){
	initMatrices();
}

void RegressionMeasurer::initMatrices(){
	meanOutputError = FeatureVector(data.getTargetSample(0,0).getLength());
	stdDevOutputError = FeatureVector(data.getTargetSample(0,0).getLength());
}

void RegressionMeasurer::processGlobalMeanOutputError(){
	totalError = 0.0;
	for(uint i= 0; i<meanOutputError.getLength();i++){
		totalError += meanOutputError[i];
	}
}

void RegressionMeasurer::measurePerformance(){
	processMeanOutputAndStdDevOutputError();
	processGlobalMeanOutputError();
}

void RegressionMeasurer::processMeanOutputAndStdDevOutputError(){
	FeatureVector output;
	ErrorVector error;
	realv oldMean;
	realv numElements=1.0;
	for(uint i=0; i < data.getNumSequences();i++){
		for(uint j=0; j < data[i].size(); j++){
			machine.forward(data[i][j]);
			output = machine.getOutputSignal();
			error = errorMeasurer.errorPerUnit(output,data.getTargetSample(i,j));
			for(uint k=0;k<output.getLength();k++){
				oldMean = meanOutputError[k];
				meanOutputError[k] = meanOutputError[k] + (error[k]-oldMean)/(numElements);
				stdDevOutputError[k] = ((numElements-1)*stdDevOutputError[k] + (error[k]-meanOutputError[k])*(error[k]-oldMean))/(numElements+1.0);
			}
			numElements = numElements + 1.0;
		}
	}
}

RegressionDataset& RegressionMeasurer::getData() const {
	return data;
}

ErrorMeasurer& RegressionMeasurer::getErrorMeasurer() const {
	return errorMeasurer;
}

FeatureVector RegressionMeasurer::getMeanOutputError() const {
	return meanOutputError;
}

void RegressionMeasurer::setMeanOutputError(FeatureVector meanOutputError) {
	this->meanOutputError = meanOutputError;
}

FeatureVector RegressionMeasurer::getStdDevOutputError() const {
	return stdDevOutputError;
}

void RegressionMeasurer::setStdDevOutputError(FeatureVector stdDevOutputError) {
	this->stdDevOutputError = stdDevOutputError;
}

realv RegressionMeasurer::getTotalError() const {
	return totalError;
}

void RegressionMeasurer::setTotalError(realv totalError) {
	this->totalError = totalError;
}

Machine& RegressionMeasurer::getMachine() const {
	return machine;
}

RegressionMeasurer::~RegressionMeasurer(){

}
