/*!
 * \file RegressionDataset.cpp
 * Body of the RegressionDataset class.
 * \author Luc Mioulet
 */

#include "RegressionDataset.hpp"

using namespace std;

RegressionDataset::RegressionDataset(string _fileName) : SupervisedDataset(_fileName){

}

int RegressionDataset::getDatasetType() const{
  return DS_REGRESSION;
}

RegressionDataset::~RegressionDataset(){

}
