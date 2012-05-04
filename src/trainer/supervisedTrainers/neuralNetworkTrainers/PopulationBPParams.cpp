/*!
 * \file PopulationBPParams.cpp
 * Body of the PopulationBPParams class.
 * \author Luc Mioulet
 */

#include "PopulationBPParams.hpp"

using namespace cv;
using namespace std;

PopulationBPParams::PopulationBPParams(realv _learningRate, realv _learningRateDecrease, uint _maxIterations, uint _maxTrained, realv _errorToFirst, realv _errorToFirstIncrease) : learningRate(_learningRate), learningRateDecrease(_learningRateDecrease), maxIterations(_maxIterations), maxTrained(_maxTrained), errorToFirst(_errorToFirst), errorToFirstIncrease(_errorToFirstIncrease){

}

realv PopulationBPParams::getLearningRate(){
  return learningRate;
}

realv PopulationBPParams::getLearningRateDecrease(){
  return learningRateDecrease;
}

uint PopulationBPParams::getMaxIterations(){
  return maxIterations;
}

uint PopulationBPParams::getMaxTrained(){
  return maxTrained;
}

realv PopulationBPParams::getErrorToFirst(){
  return errorToFirst;
}

realv PopulationBPParams::getErrorToFirstIncrease(){
  return errorToFirstIncrease;
}

void PopulationBPParams::setLearningRate(realv _learningRate){
  learningRate= _learningRate;
}

void PopulationBPParams::setLearningRateDecrease(realv _learningRateDecrease){
  learningRateDecrease= _learningRateDecrease;
}

void PopulationBPParams::setMaxIterations(uint _maxIterations){
  maxIterations= _maxIterations;
}

void PopulationBPParams::setMaxTrained(uint _maxTrained){
  maxTrained= _maxTrained;
}

void PopulationBPParams::setErrorToFirst(realv _errorToFirst){
  if(_errorToFirst<1){
    errorToFirst = _errorToFirst;
  }
  else{
    errorToFirst = 0.9999999;
  }
}

void PopulationBPParams::setErrorToFirstIncrease(realv _errorToFirstIncrease){
    errorToFirstIncrease = _errorToFirstIncrease;
}

PopulationBPParams::~PopulationBPParams(){

}
