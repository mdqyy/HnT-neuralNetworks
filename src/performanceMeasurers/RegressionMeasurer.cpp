/*!
 * \file RegressionMeasurer.cpp
 * Body of the RegressionMeasurer class.
 * \author Luc Mioulet
 */

#include "RegressionMeasurer.hpp"


RegressionMeasurer::RegressionMeasurer(NeuralNetwork& _machine, RegressionDataset& _data, ErrorMeasurer& _em) : machine(_machine), data(_data), errorMeasurer(_em){
	initMatrices();
}

void RegressionMeasurer::initMatrices(){
	meanOutputError = FeatureVector(data.getTargetSample(0,0).getLength());
	stdDevOutputError = FeatureVector(data.getTargetSample(0,0).getLength());
}

realv RegressionMeasurer::processGlobalMeanOutputError(){
	realv result = 0;
	for(uint i= 0; i<meanOutputError.getLength();i++){
		result += meanOutputError[k];
	}
	result = result/((double)meanOutputError.getLength());
	return result;
}

void RegressionMeasurer::measurePerformance(){
	processMeanOutputError();
}

void RegressionMeasurer::processMeanOutputAndStdDevOutputError(){
	FeatureVector output,error;
	realv oldMean;
	realv numElements=1.0;
	for(uint i=0; i < data.getNumSequences();i++){
		for(uint j=0; j < data[i].size(); j++){
			machine.forward(data[i][j]);
			output = machine.getOutputSignal();
			error = errorMeasurer.errorPerUnit(output,data.getTargetSample(i,j));
			for(uint k=0;k<output.getLength();k++){
				oldMean = meanOutputError[k];
				meanOutputError[k] = meanOutputError[k] + (error[k]-oldMean[k])/(numElements);
				stdDevOutputError[k] = ((numElements-1)*stdDevOutputError[k] + (error[k]-meanOutputError[k])*(error[k]-oldMean))/(numElements+1.0);
				numElements = 1.0+numElements;
			}
		}
	}
}

RegressionDataset& RegressionMeasurer::getData() const {
	return data;
}

ErrorMeasurer& RegressionMeasurer::getErrorMeasurer() const {
	return errorMeasurer;
}

Machine& RegressionMeasurer::getMachine() const {
	return machine;
}

RegressionMeasurer::~RegressionMeasurer(){

}
