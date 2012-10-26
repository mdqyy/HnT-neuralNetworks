/*!
 * \file ClassificationDataset.cpp
 * Body of the ClassificationDataset class.
 * \author Luc Mioulet
 */

#include "ClassificationDataset.hpp"

using namespace std;
using namespace cv;

ClassificationDataset::ClassificationDataset() :
		SupervisedDataset(),/* classLabels(map<int,string>()),  classLabelIndex(map<string,int>()),*/classes(vector<vector<int> >()), maxClasses(0) {
}

int ClassificationDataset::getNumberOfClasses() {
	return classLabels.size();
}

map<int, string> ClassificationDataset::getClassLabelMap() {
	return classLabels;
}

map<string, int> ClassificationDataset::getClassLabelIndexMap() {
	return classLabelIndex;
}

int ClassificationDataset::getIndexLabel(string _class) const {
	map<string, int>::const_iterator it = classLabelIndex.find(_class);
	if (it == classLabelIndex.end()) {
		throw invalid_argument("ClassificationDataset :  Class name non existent");
	}
	return it->second;
}

string ClassificationDataset::getClassLabel(int _index) const {
	map<int, string>::const_iterator it = classLabels.find(_index);
	if (it == classLabels.end()) {
		throw invalid_argument("ClassificationDataset : Class index non existent");
	}
	return it->second;
}

vector<FeatureVector> ClassificationDataset::getTargetSequence(uint _i) const {
	vector<FeatureVector> result;
	for (uint j = 0; j < classes[_i].size(); j++) {
		result.push_back(getTargetSample(_i, j));
	}
	return result;
}

FeatureVector ClassificationDataset::getTargetSample(uint _i, uint _j) const {
	int index = getSampleClassIndex(_i, _j);
	FeatureVector result(classLabels.size());
	result[index - 1] = 1.0;
	return result;
}

vector<string> ClassificationDataset::getSequenceClasses(uint _index) const {
	vector<int> temp = getSequenceClassesIndex(_index);
	vector<string> result;
	for (uint i = 0; i < temp.size(); i++) {
		result.push_back(getClassLabel(temp[i]));
	}
	return result;
}

vector<int> ClassificationDataset::getSequenceClassesIndex(uint _index) const {
	return classes[_index];
}

int ClassificationDataset::getSampleClassIndex(uint _i, uint _j) const {
	return classes[_i][_j];
}

string ClassificationDataset::getSampleClass(uint _i, uint _j) const {
	int temp = classes[_i][_j];
	return getClassLabel(temp);
}

int ClassificationDataset::getDatasetType() const {
	return DS_CLASSIFICATION;
}

void ClassificationDataset::addSequence(FeatureVector _sequence, string _class) {
	vector<FeatureVector> tempSeq;
	vector<string> tempClasses;
	tempSeq.push_back(_sequence);
	tempClasses.push_back(_class);
	addSequence(tempSeq, tempClasses);
}

void ClassificationDataset::addSequence(FeatureVector _sequence, int _class) {
	vector<FeatureVector> tempSeq;
	vector<int> tempClasses;
	tempSeq.push_back(_sequence);
	tempClasses.push_back(_class);
	addSequence(tempSeq, tempClasses);
}

void ClassificationDataset::addSequence(vector<FeatureVector> _sequence, vector<string> _classes) {
	if (_sequence.size() < _classes.size()) {
		throw length_error("ClassificationDataset : Sequence size should be greater or equal to classes size");
	}
	vector<int> indexClasses;
	for (uint i = 0; i < _classes.size(); i++) {
		if (classLabelIndex.find(_classes[i]) == classLabelIndex.end()) {
			throw invalid_argument("ClassificationDataset : Class non existent");
		}
		indexClasses.push_back(getIndexLabel(_classes[i]));
	}
	addSequence(_sequence, indexClasses);
}

void ClassificationDataset::addSequence(vector<FeatureVector> _sequence, vector<int> _classes) {
	if (_sequence.size() < _classes.size()) {
		throw length_error("ClassificationDataset : Sequence size should be greater or equal to classes size");
	}
	data.push_back(_sequence);
	classes.push_back(_classes);
	if (_sequence.size() > maxSequenceLength) {
		maxSequenceLength = _sequence.size();
	}
	if (_classes.size() > maxClasses) {
		maxClasses = _classes.size();
	}
	updateStatistics(_sequence);
}

void ClassificationDataset::addSample(FeatureVector _sample, string _className, uint _index) {
	if (_className.compare("") != 0) {
		if (classLabelIndex.find(_className) == classLabelIndex.end()) {
			throw invalid_argument("ClassificationDataset : Class non existent");
		}
	}
	addSample(_sample, getIndexLabel(_className), _index);
}

void ClassificationDataset::addSample(FeatureVector _sample, int _classIndex, uint _index) {
	if (classLabels.find(_classIndex) == classLabels.end()) {
		throw invalid_argument("ClassificationDataset : Class number not existent");
	}
	if (_index > data.size() - 1) {
		throw out_of_range("ClassificationDataset : Index out of range");
	}
	int insertIndex = _index;
	if (insertIndex < 0) { // if we are not adding data to an existing index
		data.push_back(vector<FeatureVector>(1, _sample));
		classes.push_back(vector<int>(1, _classIndex));
	} else { // else we are adding to existing data
		data[insertIndex].push_back(_sample);
		if (data[insertIndex].size() > maxSequenceLength) { //check if we de not go over the actual maximum
			maxSequenceLength = data[insertIndex].size();
		}
		if (_classIndex >= 0) {
			classes[insertIndex].push_back(_classIndex);
		}
	}
	updateStatistics(_sample);
}

void ClassificationDataset::addClass(string _class, int _index) {
	int newIndex = _index;
	if (newIndex < 0) {
		newIndex = classLabels.size() + 1;
	}
	classLabels.insert(pair<int, string>(newIndex, _class));
	classLabelIndex.insert(pair<string, int>(_class, newIndex));
}

void ClassificationDataset::load(std::string _fileName) {
	/* Open file */
	TiXmlDocument doc(_fileName);
	if (!doc.LoadFile()) {
		throw invalid_argument("ClassificationDataset : Uncorrect filename.");
	}
	TiXmlHandle hdl(&doc);
	/* Get Dataset infos */
	if (hdl.FirstChildElement("classificationDataset").Element() == 0) {
		throw invalid_argument("ClassificationDataset :This is not a classification dataset.");
	}
	TiXmlElement *elem = hdl.FirstChildElement("classificationDataset").FirstChildElement("datasetInfos").FirstChildElement("name").Element();
	name = elem->GetText();
	/* Go through the class map */
	TiXmlElement* child = hdl.FirstChild("classificationDataset").FirstChild("datasetInfos").FirstChild("classesMap").FirstChild("pair").ToElement();
	for (child; child; child = child->NextSiblingElement()) {
		int key = 0;
		if (TIXML_SUCCESS == child->QueryIntAttribute("key", &key)) {
			addClass(child->GetText(), key);
		}
	}
	/* Fill in data and classes*/
	child = hdl.FirstChild("classificationDataset").FirstChild("data").FirstChild("sequence").ToElement();
	for (child; child; child = child->NextSiblingElement()) {
		TiXmlElement* secondChild = child->FirstChild("featureVector")->ToElement();
		vector<FeatureVector> sequence;
		for (secondChild; secondChild; secondChild = secondChild->NextSiblingElement("featureVector")) {
			int length = 0;
			if (TIXML_SUCCESS != secondChild->QueryIntAttribute("length", &length)) {
				throw invalid_argument("ClassificationDataset : Uncorrect attribute name");
			}
			FeatureVector fv(length);
			stringstream ss(stringstream::in | stringstream::out);
			ss << secondChild->GetText();
			for (int n = 0; n < length; n++) {
				ss >> fv[n];
			}
			sequence.push_back(fv);
		}
		/* Read classes */
		secondChild = child->FirstChild("classes")->ToElement();
		int classLength = 0;
		if (TIXML_SUCCESS != secondChild->QueryIntAttribute("classLength", &classLength)) {
			throw invalid_argument("ClassificationDataset : Uncorrect attribute name");
		}
		vector<int> classes;
		stringstream ss(stringstream::in | stringstream::out);
		ss << secondChild->GetText();
		for (int n = 0; n < classLength; n++) {
			int value;
			ss >> value;
			classes.push_back(value);
		}
		addSequence(sequence, classes);
	}
}

void ClassificationDataset::save(std::string _fileName) {
	TiXmlDocument doc;
	TiXmlDeclaration * decl = new TiXmlDeclaration("1.0", "", "no");
	doc.LinkEndChild(decl);
	TiXmlElement * classificationDataset = new TiXmlElement("classificationDataset");
	TiXmlElement * datasetInfos = new TiXmlElement("datasetInfos");
	TiXmlElement * dname = new TiXmlElement("name");
	TiXmlText * tname = new TiXmlText(name);
	dname->LinkEndChild(tname);
	datasetInfos->LinkEndChild(dname);

	TiXmlElement * classMap = new TiXmlElement("classesMap");
	map<int, string>::iterator pos;
	for (pos = classLabels.begin(); pos != classLabels.end(); pos++) {
		TiXmlElement * dpair = new TiXmlElement("pair");
		TiXmlText * tpair = new TiXmlText(pos->second);
		dpair->LinkEndChild(tpair);
		dpair->SetAttribute("key", pos->first);
		classMap->LinkEndChild(dpair);
	}
	datasetInfos->LinkEndChild(classMap);
	classificationDataset->LinkEndChild(datasetInfos);
	TiXmlElement * ddata = new TiXmlElement("data");
	for (uint i = 0; i < data.size(); i++) {
		TiXmlElement * sequence = new TiXmlElement("sequence");
		for (uint j = 0; j < data[i].size(); j++) {
			TiXmlElement * fv = new TiXmlElement("featureVector");
			fv->SetAttribute("length", data[i][j].getLength());
			stringstream ss(stringstream::in | stringstream::out);
			for (uint k = 0; k < data[i][j].getLength(); k++) {
				ss << data[i][j][k] << " ";
			}
			TiXmlText * tfv = new TiXmlText(ss.str());
			fv->LinkEndChild(tfv);
			sequence->LinkEndChild(fv);
		}
		vector<int> clTemp = getSequenceClassesIndex(i);
		TiXmlElement * cl = new TiXmlElement("classes");
		cl->SetAttribute("classLength", clTemp.size());
		stringstream sscl(stringstream::in | stringstream::out);
		for (uint l = 0; l < clTemp.size(); l++) {
			sscl << clTemp[l] << " ";
		}
		TiXmlText * tcl = new TiXmlText(sscl.str());
		cl->LinkEndChild(tcl);
		sequence->LinkEndChild(cl);
		ddata->LinkEndChild(sequence);
	}
	classificationDataset->LinkEndChild(ddata);
	doc.LinkEndChild(classificationDataset);
	doc.SaveFile(_fileName);
}

ClassificationDataset::~ClassificationDataset() {

}

ostream& operator<<(ostream& _os, ClassificationDataset& _cd) {
	_os << "Classification dataset '" << _cd.getName() << "' : " << endl;
	_os << "\t - Class mapping  :";
	map<int, string>::iterator pos;
	for (pos = _cd.getClassLabelMap().begin(); pos != _cd.getClassLabelMap().end(); pos++) {
		_os << "(" << pos->first << ", " << pos->second << ") ;";
	}
	_os << endl;
	_os << "\t - Sequences : " << _cd.getNumSequences() << endl;
	_os << "\t - Samples : " << _cd.getNumSamples() << endl;
	for (uint i = 0; i < _cd.getNumSequences(); i++) {
		_os << "Sequence " << i << "[ " << endl;
		for (uint j = 0; j < _cd[i].size(); j++) {
			_os << _cd[i][j];
		}
		vector<int> classes = _cd.getSequenceClassesIndex(i);
		for (uint j = 0; j < classes.size(); j++) {
			_os << "\t \t With classes " << classes[j] << " ";
		}
		_os << " ]" << endl;
	}
	return _os;

}
