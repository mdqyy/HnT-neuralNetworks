/*!
 * \file ClassificationErrorMeasurer.cpp
 * Body of the ClassificationErrorMeasurer class.
 * \author Luc Mioulet
 */

#include "ClassificationErrorMeasurer.hpp"

using namespace std;
using namespace cv;

ClassificationErrorMeasurer::ClassificationErrorMeasurer(uint _numberOfExamples){

}

ErrorVector ClassificationErrorMeasurer::errorPerUnit(FeatureVector _output, FeatureVector _target){
  if(_output.getLength() != _target.getLength()){
    throw length_error("ClassificationErrorMeasurer : Output and target do not have the same size");
  }
  ErrorVector result(_output.getLength());
  realv maxOutputValue = _output[0];
  int maxOutputIndex = 0;
  for(uint i=0;i<_output.getLength();i++){
    if(_output[i]>maxOutputValue){
      maxOutputValue=_output[i];
      maxOutputIndex= i;
    }
  }
  if(_target[maxOutputIndex]!=1.0){
    result[maxOutputIndex]=1.0;
  }
  errPerUnit=result;
  return result;
}

realv ClassificationErrorMeasurer::totalError(FeatureVector _output, FeatureVector _target){
  if(_output.getLength() != _target.getLength()){
    throw length_error("ClassificationErrorMeasurer : Output and target do not have the same size");
  }
  realv result=0.0;
  realv maxOutputValue = _output[0];
  int maxOutputIndex = 0;
  for(uint i=0;i<_output.getLength();i++){
    if(_output[i]>maxOutputValue){
      maxOutputValue=_output[i];
      maxOutputIndex= i;
    }
  }
  if(_target[maxOutputIndex]!=1){
    result=1.0;
  }
  error=result;
  return result;
}

ClassificationErrorMeasurer::~ClassificationErrorMeasurer(){

}
