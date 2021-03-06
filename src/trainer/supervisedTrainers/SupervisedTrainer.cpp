/*!
 * \file SupervisedTrainer.cpp
 * Body of the Trainer class.
 * \author Luc Mioulet
 */

#include "SupervisedTrainer.hpp"

using namespace std;
using namespace cv;

SupervisedTrainer::SupervisedTrainer(Machine& _machine, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask, ostream& _log ) : Trainer(_machine, _data, _featureMask, _indexMask, _log), trainData(_data){

}

vector<uint> SupervisedTrainer::defineIndexOrderSelection(uint _numSequences){
  vector<uint> indexOrder;
  for(uint i=0 ;  i<_numSequences; i++){
    indexOrder.push_back(i);
  }
  int exchangeIndex=0;
  RNG random(getTickCount());
  random.next();
  for(uint i=0 ;  i<_numSequences; i++){	 
    random.next();
    exchangeIndex=random.uniform(0,_numSequences);
    swap(indexOrder[i],indexOrder[exchangeIndex]);
  } 
  return indexOrder;
}

SupervisedTrainer::~SupervisedTrainer(){

}
