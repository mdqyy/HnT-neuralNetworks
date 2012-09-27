/*!
 * \file AEMeasurer.cpp
 * Body of the AEMeasurer class.
 * \author Luc Mioulet
 */

#include "AEMeasurer.hpp"

using namespace std;

AEMeasurer::AEMeasurer() {

}

void AEMeasurer::processErrors(FeatureVector _output, FeatureVector _target) {
	if (_output.getLength() != _target.getLength()) {
		throw length_error("AEMeasurer : Output and target do not have the same size");
	}
	ErrorVector result(_output.getLength());
	err = 0;
	for (uint i = 0; i < _output.getLength(); i++) {
		result[i] = abs(_output[i] - _target[i]);
		err += result[i];
	}
	errPerUnit = result;
}

ErrorVector AEMeasurer::errorPerUnit(FeatureVector _output, FeatureVector _target) {
	if (_output.getLength() != _target.getLength()) {
		throw length_error("AEMeasurer : Output and target do not have the same size");
	}
	ErrorVector result(_output.getLength());
	for (uint i = 0; i < _output.getLength(); i++) {
		result[i] = abs(_output[i] - _target[i]);
	}
	errPerUnit = result;
	return result;
}

realv AEMeasurer::totalError(FeatureVector _output, FeatureVector _target) {
	if (_output.getLength() != _target.getLength()) {
		throw length_error("AEMeasurer : Output and target do not have the same size");
	}
	realv result = 0;
	for (uint i = 0; i < _output.getLength(); i++) {
		result += abs(_output[i] - _target[i]);
	}
	err = result ;
	return err;
}

AEMeasurer::~AEMeasurer() {

}

ostream& operator<<(ostream& _os, AEMeasurer& _ae) {
	_os << _ae.getError();
	return _os;
}
