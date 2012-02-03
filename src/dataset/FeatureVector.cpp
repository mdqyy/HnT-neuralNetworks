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
