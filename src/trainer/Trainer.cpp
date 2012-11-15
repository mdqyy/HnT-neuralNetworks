/*!
 * \file Trainer.cpp
 * Body of the Trainer class.
 * \author Luc Mioulet
 */

#include "Trainer.hpp"

Trainer::Trainer(Machine& _machine, Dataset& _data, Mask& _featureMask, Mask& _indexMask) :
		machine(_machine), data(_data), featureMask(_featureMask), indexMask(_indexMask) {
	assert(featureMask.getLength()==0 || featureMask.getLength()==data.getFeatureVectorLength());
	assert(indexMask.getLength()==0 || indexMask.getLength()==data.getNumSequences());
}

Dataset& Trainer::getTrainDataset() const {
	return data;
}

Machine& Trainer::getMachine() const {
	return machine;
}

Mask& Trainer::getFeatureMask() const {
	return featureMask;
}

Mask& Trainer::getIndexMask() const {
	return indexMask;
}

Trainer::~Trainer() {

}
