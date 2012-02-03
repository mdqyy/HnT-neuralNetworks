/*!
 * \file Dataset.cpp
 * Body of the Dataset class.
 * \author Luc Mioulet
 */

#include "Dataset.hpp"

using namespace std;
using namespace cv;

Dataset::Dataset(string _file) :name(""), 
  data(vector< vector<FeatureVector> >()),
  numSamples(0),
  numSequences(0),
  maxSequenceLength(0),
  fvLength(0),
  meanMat(0),
  Qmat(0){

}

std::string Dataset::getName() const{
  return name;
}

std::vector< std::vector<FeatureVector> > Dataset::getData() const{
  return data;
}

uint Dataset::getNumSamples() const{
  return numSamples;
}

uint Dataset::getNumSequences() const{
  return numSequences;
}

uint Dataset::getMaxSequenceLength() const{
  return maxSequenceLength;
}

uint Dataset::getFeatureVectorLength() const{
  return fvLength;
}

ValueVector Dataset::getMean() const{
  return meanMat;
}

ValueVector Dataset::getStandardDeviation() const{
  ValueVector stdevTemp(fvLength);
  for (uint i=0;i<fvLength;i++){
    stdevTemp[i]=sqrt(Qmat[i]/((realv)numSamples));
  }
  return stdevTemp;
}

vector<FeatureVector>& Dataset::operator[](uint _index){
  if(_index > data.size()-1){
    throw out_of_range("Out of range index");
  }
  return data[_index];
}

const vector<FeatureVector>& Dataset::operator[](uint _index) const{
  if(_index > data.size()-1){
    throw out_of_range("Out of range index");
  }
  return data[_index];
}

void Dataset::updateStatistics(FeatureVector _sample){
  numSamples+=1;
  if(_sample.getLength()!=fvLength){
    throw length_error("Wrong sample size");
  }
  for(uint i=0;i<fvLength;i++){
    realv oldMean=meanMat[i];
    meanMat[i] = oldMean + (_sample[i]-oldMean)/((realv)numSamples);
    Qmat[i] = Qmat[i] + (_sample[i]-oldMean)*(_sample[i]-meanMat[i]);
  }
}

void Dataset::updateStatistics(std::vector<FeatureVector> _sequence){
  numSequences+=1;
  for(uint i=0;i<_sequence.size();i++){
    updateStatistics(_sequence[i]);
  }
}

Dataset::~Dataset(){

}
