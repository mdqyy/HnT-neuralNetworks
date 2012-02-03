/*!
 * \file ValueVector.cpp
 * Body of the ValueVector class.
 * \author Luc Mioulet
 */

#include "ValueVector.hpp"

using namespace cv;
using namespace std;

ValueVector::ValueVector(int length){
  #ifdef REAL_DOUBLE
  data = Mat(length,1,CV_64FC1,0.0);
  #else
  data = Mat(length,1CV_32FC1,0.0);
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

const realv& ValueVector::operator[](int i) const{
  if(i<0 || i>data.rows){
    throw out_of_range("out of range in vector access \n");
  }
  #ifdef REAL_DOUBLE
  return data.at<double>(i,0);
  #else 
  return data.at<float>(i,0);
  #endif
}

realv& ValueVector::operator[](int i){
  if(i<0 || i>data.rows){
    throw out_of_range("out of range in vector access \n");
  }
  #ifdef REAL_DOUBLE
  return data.at<double>(i,0);
  #else 
  return data.at<float>(i,0);
  #endif
}

void ValueVector::reset(realv _default){
  #ifdef REAL_DOUBLE
  data = Mat(getLength(),1,CV_64FC1,_default);
  #else
  data = Mat(getLength(),1CV_32FC1,_default);
  #endif
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
