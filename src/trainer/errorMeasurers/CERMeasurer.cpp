/*!
 * \file CERMeasurer.cpp
 * Body of the WECMeasurer class. Character error rate between a target and sample sequence.
 * \author Luc Mioulet
 */

#include "CERMeasurer.hpp"

CERMeasurer::CERMeasurer() {

}

void CERMeasurer::processErrors(FeatureVector _output, FeatureVector _target){

}

realv CERMeasurer::totalError(vector<int> _reference, vector<int> _output){
	realv result = 0;
	int i = 0;
	result = (realv)abs(_reference.size() - output.size());
	while(i < _reference.size() && i < _output.size()){
		if(_reference[i]!=_output[i]){
			result +=1.0;
		}
	}
	result = result/((realv)_reference.size());
	err = result
	return result;
}

ErrorVector CERMeasurer::errorPerUnit(FeatureVector _output, FeatureVector _target){
	ErrorVector result(_output.getLength());
	errPerUnit = result;
	return result;
}

realv CERMeasurer::totalError(FeatureVector _output, FeatureVector _target){
	realv result=0;
	err = result ;
	return err;
}

CERMeasurer::~CERMeasurer() {

}

