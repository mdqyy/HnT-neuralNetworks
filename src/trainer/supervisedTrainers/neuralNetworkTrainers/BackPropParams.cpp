/*!
 * \file BackPropParams.cpp
 * Body of the BackPropParams class.
 * \author Luc Mioulet
 */

#include "BackPropParams.hpp"

using namespace cv;
using namespace std;

BackPropParams::BackPropParams(bool _doStochastic,realv _learningRate, realv _learningRateDecrease, uint _maxIterations, realv _minError, realv _minChangeError) : doStochastic(_doStochastic), learningRate(_learningRate), learningRateDecrease(_learningRateDecrease), maxIterations(_maxIterations), minError(_minError), minChangeError(_minChangeError){

}

bool BackPropParams::getDoStochastic(){
  return doStochastic;
}

realv BackPropParams::getLearningRate(){
  return learningRate;
}

realv BackPropParams::getLearningRateDecrease(){
  return learningRateDecrease;
}

uint BackPropParams::getMaxIterations(){
  return maxIterations;
}

realv BackPropParams::getMinError(){
  return minError;
}

realv BackPropParams::getMinChangeError(){
  return minChangeError;
}

void BackPropParams::setDoStochastic(bool _doStochastic){
  doStochastic= _doStochastic;
}

void BackPropParams::setLearningRate(realv _learningRate){
  learningRate= _learningRate;
}

void BackPropParams::setLearningRateDecrease(realv _learningRateDecrease){
  learningRateDecrease= _learningRateDecrease;
}

void BackPropParams::setMaxIterations(uint _maxIterations){
  maxIterations= _maxIterations;
}

void BackPropParams::setMinError(realv _minError){
  minError= _minError;
}

void BackPropParams::setMinChangeError(realv _minChangeError){
  minChangeError = _minChangeError;
}

BackPropParams::~BackPropParams(){

}
