/*!
 * \file ErrorVector.cpp
 * Body of the ErrorVector class.
 * \author Luc Mioulet
 */

#include "ErrorVector.hpp"

using namespace cv;
using namespace std;

ErrorVector::ErrorVector(int length) : ValueVector(length){

}

ErrorVector::ErrorVector(Mat _data) : ValueVector(_data){

}

ErrorVector::~ErrorVector(){

}

ErrorVector::ErrorVector(vector<ErrorVector> _vecs) {
  uint length = 0;
  for(uint i=0;i<_vecs.size();i++){
    length += _vecs[i].getLength();
  }
  #ifdef REAL_DOUBLE
  data = Mat(length,1,CV_64FC1,0.0);
  #else
  data = Mat(length,1,CV_32FC1,0.0);
  #endif
  int index = 0;
  for(uint i=0;i<_vecs.size();i++){
    for(uint j=0;j<_vecs[i].getLength();j++){
      data.at<realv>(index,0)= _vecs[i][j];
      index++;
    }
  }
}

ostream& operator<<(ostream& os, const ErrorVector& fv){
  os << "Error vector : " << endl;
  os << "\t -length :" << fv.getLength() << endl;
  os << "\t -containing : ";
  for (int i = 0; i < fv.getLength(); i++){
    os << fv[i] <<" ; " ;
    }
  os << endl;
  return os;
}

ofstream& operator<<(ofstream& ofs, const ErrorVector& ev){
  ofs << "< L " << ev.getLength() << " [";
  for (int i = 0; i < ev.getLength(); i++){
    ofs << " "<< ev[i];
    }
  ofs << "] > "<< endl;
  return ofs;

  return ofs;
}

ifstream& operator>>(ifstream& ifs, ErrorVector& ev){
  uint vLength;
  realv value;
  string temp;
  ifs >> temp >> vLength >> temp;
  ev= ErrorVector(vLength);
  for (int i = 0; i < ev.getLength(); i++){
    ifs >> value ;
    ev[i]= value;
  }
  ifs >> temp;
  return ifs;
}
