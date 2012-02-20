/*!
 * \file SupervisedTrainer.cpp
 * Body of the Trainer class.
 * \author Luc Mioulet
 */

#include "SupervisedTrainer.hpp"

SupervisedTrainer::SupervisedTrainer(Machine& _machine, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask ) : Trainer(_machine, _data, _featureMask, _indexMask), trainData(_data){

}

SupervisedTrainer::~SupervisedTrainer(){

}
