/*!
 * \file SequenceClassDataset.cpp
 * Body of the SequenceClassDataset class.
 * \author Luc Mioulet
 */

#include "SequenceClassDataset.hpp"

using namespace std;
using namespace cv;

SequenceClassDataset::SequenceClassDataset() : classLabels(map<int,string>()),  classLabelIndex(map<string,int>()), classes(vector<vector<int> >()), maxClasses(0) {
}

SequenceClassDataset::SequenceClassDataset(map<int, std::string> _classLabels, map<std::string, int> _classLabelIndex, vector<vector<int> > _classes, uint _maxClasses) : classLabels(_classLabels) , classLabelIndex(_classLabelIndex), classes(_classes), maxClasses(_maxClasses){

}

int SequenceClassDataset::getNumberOfClasses() const{
	return classLabels.size();
}

map<int, string> SequenceClassDataset::getClassLabelMap() const{
	return classLabels;
}

map<string, int> SequenceClassDataset::getClassLabelIndexMap() const{
	return classLabelIndex;
}

int SequenceClassDataset::getIndexLabel(string _class) const {
	map<string, int>::const_iterator it = classLabelIndex.find(_class);
	if (it == classLabelIndex.end()) {
		throw invalid_argument("SequenceClassDataset :  Class name non existent");
	}
	return it->second;
}

string SequenceClassDataset::getClassLabel(int _index) const {
	map<int, string>::const_iterator it = classLabels.find(_index);
	if (it == classLabels.end()) {
		throw invalid_argument("SequenceClassDataset : Class index non existent");
	}
	return it->second;
}

vector<FeatureVector> SequenceClassDataset::getTargetSequence(uint _i) const {
	vector<FeatureVector> result;
	for (uint j = 0; j < classes[_i].size(); j++) {
		result.push_back(getTargetSample(_i, j));
	}
	return result;
}

FeatureVector SequenceClassDataset::getTargetSample(uint _i, uint _j) const {
	int index = getSampleClassIndex(_i, _j);
	FeatureVector result(classLabels.size());
	result[index - 1] = 1.0;
	return result;
}

vector<string> SequenceClassDataset::getSequenceClasses(uint _index) const {
	vector<int> temp = getSequenceClassesIndex(_index);
	vector<string> result;
	for (uint i = 0; i < temp.size(); i++) {
		result.push_back(getClassLabel(temp[i]));
	}
	return result;
}

vector<int> SequenceClassDataset::getSequenceClassesIndex(uint _index) const {
	return classes[_index];
}

int SequenceClassDataset::getSampleClassIndex(uint _i, uint _j) const {
	return classes[_i][_j];
}

string SequenceClassDataset::getSampleClass(uint _i, uint _j) const {
	int temp = classes[_i][_j];
	return getClassLabel(temp);
}

void SequenceClassDataset::addSequence(string _class) {
	vector<string> tempClasses;
	tempClasses.push_back(_class);
	addSequence(tempClasses);
}

void SequenceClassDataset::addSequence(int _class) {
	vector<int> tempClasses;
	tempClasses.push_back(_class);
	addSequence(tempClasses);
}

void SequenceClassDataset::addSequence(vector<string> _classes) {
	vector<int> indexClasses;
	for (uint i = 0; i < _classes.size(); i++) {
		if (classLabelIndex.find(_classes[i]) == classLabelIndex.end()) {
		  cout <<_classes[i] << endl;
			throw invalid_argument("SequenceClassDataset : Class non existent");
		}
		indexClasses.push_back(getIndexLabel(_classes[i]));
	}
	addSequence(indexClasses);
}

void SequenceClassDataset::addSequence(vector<int> _classes) {
	classes.push_back(_classes);
	if (_classes.size() > maxClasses) {
		maxClasses = _classes.size();
	}
}

void SequenceClassDataset::addSample(string _className, uint _index) {
	if (_className.compare("") != 0) {
		if (classLabelIndex.find(_className) == classLabelIndex.end()) {
			throw invalid_argument("SequenceClassDataset : Class non existent");
		}
	}
	addSample(getIndexLabel(_className), _index);
}

void SequenceClassDataset::addSample(int _classIndex, uint _index) {
	if (classLabels.find(_classIndex) == classLabels.end()) {
		throw invalid_argument("SequenceClassDataset : Class number not existent");
	}
	int insertIndex = _index;
	if (insertIndex < 0) { // if we are not adding data to an existing index
		classes.push_back(vector<int>(1, _classIndex));
	}
	else { // else we are adding to existing data
		if (_classIndex >= 0) {
			classes[insertIndex].push_back(_classIndex);
		}
	}
}

void SequenceClassDataset::addClass(string _class, int _index) {
	int newIndex = _index;
	if (newIndex < 0) {
		newIndex = classLabels.size() + 1;
	}
	classLabels.insert(pair<int, string>(newIndex, _class));
	classLabelIndex.insert(pair<string, int>(_class, newIndex));
}

void SequenceClassDataset::load(std::string _fileName) {
  ifstream ifs;
    ifs.open(_fileName.c_str());
    int length,number;
    string letter;
    ifs >> length;
    for(int i= 0 ; i < length; i++){
      ifs >> number ;
      ifs >> letter;
      addClass(letter,number);
    }
    while(!ifs.eof()){
      ifs >> length;
      vector<int> vec;
      for(int i= 0 ; i < length; i++){
        ifs >> number;
        vec.push_back(number);
      }
      addSequence(vec);
    }

}

void SequenceClassDataset::save(std::string _fileName) {
  ofstream ofs;
  ofs.open(_fileName.c_str());
  ofs << classLabels.size() << endl;
  for(map<int,string>::iterator it=classLabels.begin(); it!=classLabels.end();it++){
    ofs << (*it).first << " " << (*it).second << endl;
  }
  for(uint i=0;i< classes.size();i++){
    ofs << classes[i].size() << " ";
    for(uint j=0; j<classes[i].size();j++){
      ofs << classes[i][j] << " ";
    }
    ofs << endl;
  }
}

SequenceClassDataset::~SequenceClassDataset() {

}

std::vector<std::vector<int> > SequenceClassDataset::getClasses() const {
	return classes;
}

std::map<std::string, int> SequenceClassDataset::getClassLabelIndex() const {
	return classLabelIndex;
}

void SequenceClassDataset::setClassLabelIndex(std::map<std::string, int> classLabelIndex) {
	this->classLabelIndex = classLabelIndex;
}

std::map<int, std::string> SequenceClassDataset::getClassLabels() const {
	return classLabels;
}

void SequenceClassDataset::setClassLabels(std::map<int, std::string> classLabels) {
	this->classLabels = classLabels;
}

uint SequenceClassDataset::getMaxClasses() const {
	return maxClasses;
}

ostream& operator<<(ostream& _os, SequenceClassDataset& _cd) {
  return _os;
}
