/*!
 * \file Trainer.cpp
 * Body of the Trainer class.
 * \author Luc Mioulet
 */

#include "Trainer.hpp"

Trainer::Trainer(Machine& _machine, Dataset& _data) : machine(_machine), data(_data){

}

Dataset& Trainer::getTrainDataset() const{
  return data;
}

Machine& Trainer::getMachine() const{
  return machine;
}

Trainer::~Trainer(){

}
