/*!
 * \file ErrorMeasurer.cpp
 * Body of the ErrorMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"

ErrorMeasurer::ErrorMeasurer() : error(0.0), errPerUnit(ErrorVector(0)){

}

realv ErrorMeasurer::getError(){
  return error;
}

ErrorVector ErrorMeasurer::getErrorPerUnit(){
  return errPerUnit;
}

void ErrorMeasurer::setError(realv _error){
  error=_error;
}

void ErrorMeasurer::setErrorPerUnit(ErrorVector _errPerUnit){
  errPerUnit = _errPerUnit;
}

ErrorMeasurer::~ErrorMeasurer(){

}
