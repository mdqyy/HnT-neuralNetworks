/*!
 * \file BackPropParams.cpp
 * Body of the BackPropParams class.
 * \author Luc Mioulet
 */

#include "BackPropParams.hpp"

BackPropParams::BackPropParams(bool _doStochastic,realv _learningRate, realv _learningRateDecrease) : doStochastic(_doStochastic), learningRate(_learningRate), learningRateDecrease(_learningRateDecrease){

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

void BackPropParams::setDoStochastic(bool _doStochastic){
  doStochastic= _doStochastic;
}

void BackPropParams::setLearningRate(realv _learningRate){
  learningRate= _learningRate;
}

void BackPropParams::setLearningRateDecrease(realv _learningRateDecrease){
  learningRateDecrease= _learningRateDecrease;
}

BackPropParams::~BackPropParams(){

}
