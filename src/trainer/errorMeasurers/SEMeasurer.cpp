/*!
 * \file SEMeasurer.cpp
 * Body of the SEMeasurer class.
 * \author Luc Mioulet
 */

#include "SEMeasurer.hpp"

using namespace std;

SEMeasurer::SEMeasurer() {

}

void SEMeasurer::processErrors(FeatureVector _output, FeatureVector _target) {
	if (_output.getLength() != _target.getLength()) {
		throw length_error("SEMeasurer : Output and target do not have the same size");
	}
	ErrorVector result(_output.getLength());
	err = 0;
	for (uint i = 0; i < _output.getLength(); i++) {
		result[i] = (_output[i] - _target[i]) * (_output[i] - _target[i]);
		err += result[i];
	}
	errPerUnit = result;
}

ErrorVector SEMeasurer::errorPerUnit(FeatureVector _output, FeatureVector _target) {
	if (_output.getLength() != _target.getLength()) {
		throw length_error("SEMeasurer : Output and target do not have the same size");
	}
	ErrorVector result(_output.getLength());
	for (uint i = 0; i < _output.getLength(); i++) {
		result[i] = (_output[i] - _target[i]) * (_output[i] - _target[i]);
	}
	errPerUnit = result;
	return result;
}

realv SEMeasurer::totalError(FeatureVector _output, FeatureVector _target) {
	if (_output.getLength() != _target.getLength()) {
		throw length_error("SE : Output and target do not have the same size");
	}
	realv result = 0;
	for (uint i = 0; i < _output.getLength(); i++) {
		result += (_output[i] - _target[i]) * (_output[i] - _target[i]);
	}
	err = result;
	return err;
}

SEMeasurer::~SEMeasurer() {

}

ostream& operator<<(ostream& _os, SEMeasurer& _se) {
	_os << _se.getError();
	return _os;
}
