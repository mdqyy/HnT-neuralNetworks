/*!
 * \file ClassDataset.cpp
 * Body of the ClassDataset class.
 * \author Luc Mioulet
 */

#include "ClassDataset.hpp"

using namespace std;
using namespace cv;

ClassDataset::ClassDataset() : classes(vector<uint>()), numberOfClasses(0) {
}

ClassDataset::ClassDataset(vector< uint > _classes, uint _numberOfClasses) : classes(_classes) ,  numberOfClasses(_numberOfClasses){
}

void ClassDataset::addClass(uint _class) {
  if(_class < numberOfClasses){
    classes.push_back(_class);
  }
  else{
    cerr << " you tried to add a non authorized class" << endl;
  }
}

FeatureVector ClassDataset::getFeatureVector(uint _index){
   if(_index >= classes.size()){
      cerr << "You tried to access a non existing index" << endl;
   }
   FeatureVector fv(numberOfClasses);
   uint classValue = classes[_index];
   fv[classValue] = 1.0;
   return fv;
}

uint ClassDataset::getClass(uint _index) const{
  if(_index >= classes.size()){
    cerr << "You tried to access a non existing index" << endl;
  }
  return classes[_index];
}

uint ClassDataset::getClassesLength() const{
  return classes.size();
}

vector<uint> ClassDataset::getClasses() const {
  return classes;
}

void ClassDataset::setClasses(vector<uint> _classes) {
  classes = _classes;
}

uint ClassDataset::getNumberOfClasses() const {
  return numberOfClasses;
}

void ClassDataset::setNumberOfClasses(uint _numClasses) {
   numberOfClasses = _numClasses;
}

void ClassDataset::load(string _fileName){
  ifstream ifs;
  ifs.open(_fileName.c_str());
  uint number;
  ifs >> number;
  setNumberOfClasses(number);
  while(!ifs.eof()){
    ifs >> number;
    addClass(number);
  }
}

void ClassDataset::save(string _fileName){
  ofstream ofs;
  ofs.open(_fileName.c_str());
  ofs << numberOfClasses << endl;
  for(uint i=0;i<classes.size();i++){
    ofs << classes[i] << endl;
  }
}


ClassDataset::~ClassDataset(){

}

ostream& operator<<(ostream& _os, ClassDataset& _cd){
    _os << "Class dataset containing " <<_cd.getNumberOfClasses() << " images, and of length " << _cd.getClassesLength()<< endl;
    return _os;
}
