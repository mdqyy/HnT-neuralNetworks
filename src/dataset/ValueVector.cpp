/*!
 * \file ValueVector.cpp
 * Body of the ValueVector class.
 * \author Luc Mioulet
 */

#include "ValueVector.hpp"

using namespace cv;
using namespace std;

ValueVector::ValueVector(int _length){
  #ifdef REAL_DOUBLE
  data = Mat(_length,1,CV_64FC1,0.0);
  #else
  data = Mat(_length,1CV_32FC1,0.0);
  #endif
}

ValueVector::ValueVector(Mat _data) : data(_data){
  #ifdef REAL_DOUBLE
  assert(data.type()==CV_64FC1);
  #else
  assert(data.type()==CV_32FC1);
  #endif
  assert(data.rows>0 && data.cols==1);
}

uint ValueVector::getLength() const{
  return (uint)data.rows;
}

const realv& ValueVector::operator[](int _index) const{
  if(_index<0 || _index>=data.rows){
    throw out_of_range("Value Vector : Out of range in vector access \n");
  }
  #ifdef REAL_DOUBLE
  return data.at<double>(_index,0);
  #else 
  return data.at<float>(_index,0);
  #endif
}

realv& ValueVector::operator[](int _index){
  if(_index<0 || _index>=data.rows){
    throw out_of_range("Value Vector : Out of range in vector access \n");
  }
  #ifdef REAL_DOUBLE
  return data.at<double>(_index,0);
  #else 
  return data.at<float>(_index,0);
  #endif
}

void ValueVector::reset(realv _default){
  #ifdef REAL_DOUBLE
  data = Mat(getLength(),1,CV_64FC1,_default);
  #else
  data = Mat(getLength(),1CV_32FC1,_default);
  #endif
}

void ValueVector::getMin(realv *_min,int *_minLoc){
  int minIndex=0;
  realv minValue=this->operator[](0);
  for(uint i=0;i<getLength();i++){
    if(this->operator[](i)<minValue){
      minValue=this->operator[](i);
      minIndex=i;
    }
  }
  *_min=minValue;
  *_minLoc=minIndex;
}

void ValueVector::getMax(realv *_max, int *_maxLoc){
  int maxIndex=0;
  realv maxValue=this->operator[](0);
  for(uint i=0;i<getLength();i++){
    if(this->operator[](i)>maxValue){
      maxValue=this->operator[](i);
      maxIndex=i;
    }
  } 
  *_max= maxValue;
  *_maxLoc= maxIndex;
}

ValueVector::~ValueVector(){

}

ostream& operator<<(ostream& os, const ValueVector& fv){
  os << "Value vector : " << endl;
  os << "\t -length :" << fv.getLength() << endl;
  os << "\t -containing : ";
  for (int i = 0; i < fv.getLength(); i++){
    os << fv[i] <<" ; " ;
    }
  os << endl;
  return os;
}
