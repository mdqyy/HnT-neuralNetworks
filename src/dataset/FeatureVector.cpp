/*!
 * \file FeatureVector.cpp
 * Body of the FeatureVector class.
 * \author Luc Mioulet
 */

#include "FeatureVector.hpp"

using namespace cv;
using namespace std;

FeatureVector::FeatureVector(int length) : ValueVector(length){

}

FeatureVector::FeatureVector(Mat _data) : ValueVector(_data){

}

FeatureVector::FeatureVector(vector<FeatureVector> _vecs){
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

FeatureVector::~FeatureVector(){

}

ostream& operator<<(ostream& os, const FeatureVector& fv){
  os << "Feature vector : " << endl;
  os << "\t -length :" << fv.getLength() << endl;
  os << "\t -containing : ";
  for (int i = 0; i < fv.getLength(); i++){
    os << fv[i] <<" ; " ;
    }
  os << endl;
  return os;
}

ofstream& operator<<(ofstream& ofs, const FeatureVector& fv){
  ofs << "< L " << fv.getLength() << " [";
  for (int i = 0; i < fv.getLength(); i++){
    ofs << " "<< fv[i];
    }
  ofs << "] > "<< endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, FeatureVector& fv){
  uint vLength;
  realv value;
  string temp;
  ifs >> temp >> vLength >> temp;
  fv= FeatureVector(vLength);
  for (int i = 0; i < fv.getLength(); i++){
    ifs >> value ;
    fv[i]= value;
  }
  ifs >> temp;
  return ifs;
}
