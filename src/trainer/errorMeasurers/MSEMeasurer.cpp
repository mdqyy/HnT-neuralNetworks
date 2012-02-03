/*!
 * \file MSEMeasurer.cpp
 * Body of the MSEMeasurer class.
 * \author Luc Mioulet
 */

#include "MSEMeasurer.hpp"

using namespace std;

MSEMeasurer::MSEMeasurer(){

}

ErrorVector MSEMeasurer::errorPerUnit(FeatureVector _output, FeatureVector _target){
  if(_output.getLength() != _target.getLength()){
    throw length_error("Output and target do not have the same size");
  }
  ErrorVector result(_output.getLength());
  for(uint i=0; i<_output.getLength();i++){
    result[i]=(_output[i] - _target[i])*(_output[i] - _target[i]);
  }
  return result;
}

realv MSEMeasurer::totalError(FeatureVector _output, FeatureVector _target){
  if(_output.getLength() != _target.getLength()){
    throw length_error("Output and target do not have the same size");
  }
  realv result=0;
  for(uint i=0; i<_output.getLength();i++){
    result+=(_output[i] - _target[i])*(_output[i] - _target[i]);
  }
  return result;
}

MSEMeasurer::~MSEMeasurer(){
  
}
