/*!
 * \file SupervisedTrainer.cpp
 * Body of the Trainer class.
 * \author Luc Mioulet
 */

#include "SupervisedTrainer.hpp"

SupervisedTrainer::SupervisedTrainer(Machine& _machine, SupervisedDataset& _data, CrossValidationParams& _cvParams) : Trainer(_machine,_data), trainData(_data),validationData(_data),testData(_data),cvParams(_cvParams){

}

SupervisedTrainer::SupervisedTrainer(Machine& _machine, SupervisedDataset& _trainData, SupervisedDataset& _validationData, SupervisedDataset& _testData, CrossValidationParams& _cvParams) : Trainer(_machine,_trainData), trainData(_trainData), validationData(_validationData), testData(_testData), cvParams(_cvParams){

}

SupervisedDataset& SupervisedTrainer::getValidationDataset() const{
  return validationData;
}

SupervisedDataset& SupervisedTrainer::getTestDataset() const{
  return testData;
}

CrossValidationParams& SupervisedTrainer::getCrossValidationParams() const{
  return cvParams;
}

SupervisedTrainer::~SupervisedTrainer(){

}
