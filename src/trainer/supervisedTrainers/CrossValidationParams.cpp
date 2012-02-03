/*!
 * \file CrossValidationParams.cpp
 * Body of the CrossValidationParams class.
 * \author Luc Mioulet
 */

#include "CrossValidationParams.hpp"

using namespace cv;

CrossValidationParams::CrossValidationParams() : doCV(false), crossValidationType(NOCV), numberOfFolds(0){
  
}

CrossValidationParams::CrossValidationParams(int _crossValidationType, int _numberOfFolds) : doCV(true), crossValidationType(_crossValidationType), numberOfFolds(_numberOfFolds){

}

bool CrossValidationParams::doCrossValidation() const{
  return doCV;
}

int CrossValidationParams::getCrossValidationType() const{
  return crossValidationType;
}

uint CrossValidationParams::getNumberOfFolds() const{
  return numberOfFolds;
}

CrossValidationParams::~CrossValidationParams(){

}
