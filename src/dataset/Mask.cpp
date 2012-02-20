/*!
 * \file Mask.cpp
 * Body of the Mask class.
 * \author Luc Mioulet
 */

#include "Mask.hpp"

using namespace std;
using namespace cv;

Mask::Mask(uint _length){
  for(uint i=0;i<_length;i++){
    mask.push_back(true);
  }
}

Mask::Mask(Mat _mask) {
  assert(_mask.cols==1 || _mask.rows==1);
  bool colonMask = (_mask.cols==1);
  if(colonMask){
    for(int i=0; i<_mask.cols;i++){
      mask.push_back(_mask.at<uchar>(0,i)==1);
    }
  }
  else{
    for(int i=0; i<_mask.rows;i++){
      mask.push_back(_mask.at<uchar>(i,0)==1);
    }
  }
}

uint Mask::getLength(){
  return mask.size();
}

const bool& Mask::operator[](uint _index) const{
  if(_index<0 || _index>=mask.size()){
    throw out_of_range("Mask : Out of range in vector access \n");
  }
 return mask[_index];
}

bool Mask::operator[](uint _index){
  if(_index<0 || _index>mask.size()){
    throw out_of_range("Mask : Out of range in vector access \n");
  }
  return mask[_index];
}

void Mask::reset(bool _default){
  for(int i=0;i<mask.size();i++){
    mask[i]=_default;
  }
}

Mask::~Mask(){

}
