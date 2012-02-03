/*!
 * \file UnsupervisedDataset.cpp
 * Body of the UnsupervisedDataset class.
 * \author Luc Mioulet
 */

#include "UnsupervisedDataset.hpp"

using namespace std;

UnsupervisedDataset::UnsupervisedDataset(string _fileName): Dataset(_fileName){

}

void UnsupervisedDataset::addSequence(vector<FeatureVector> sequence){
  data.push_back(sequence);
  if( sequence.size() > maxSequenceLength){
    maxSequenceLength = sequence.size();
  }
  updateStatistics(sequence);
}

void UnsupervisedDataset::addSample(FeatureVector sample, int index){
  int insertIndex=index;
  if(insertIndex<0){
    data.push_back(vector<FeatureVector>(1,sample));
  }
  else{
    data[index].push_back(sample);
    if( data[index].size() > maxSequenceLength){ //check if we de not go over the actual maximum 
      maxSequenceLength = data[index].size(); 
    }
  }
  updateStatistics(sample);
}

UnsupervisedDataset::~UnsupervisedDataset(){

}
