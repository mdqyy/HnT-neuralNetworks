/*!
 * \file RegressionDataset.cpp
 * Body of the RegressionDataset class.
 * \author Luc Mioulet
 */

#include "RegressionDataset.hpp"

using namespace std;
using namespace cv;

RegressionDataset::RegressionDataset() : SupervisedDataset(), values(vector <vector<FeatureVector> >()){

}

int RegressionDataset::getDatasetType() const{
  return DS_REGRESSION;
}

void RegressionDataset::addSequence(vector<FeatureVector> _sequence, vector<FeatureVector> _value){
  data.push_back(_sequence);
  values.push_back(_value);
  updateStatistics(_sequence);
}

void RegressionDataset::load(std::string _fileName){
  /* Open file */
  TiXmlDocument doc( _fileName );
  if ( !doc.LoadFile() ){
    throw invalid_argument("RegressionDataset : Uncorrect filename");
  }
  TiXmlHandle hdl(&doc);
  /* Get Dataset infos */
  TiXmlElement *elem = hdl.FirstChildElement("regressionDataset").FirstChildElement("datasetInfos").FirstChildElement("name").Element();
  name = elem->GetText();
  /* Fill in the feature vectors */
  TiXmlElement* child = hdl.FirstChild( "regressionDataset" ).FirstChild( "data" ).FirstChild( "sequence" ).ToElement();  
  for(child; child ;child = child->NextSiblingElement()){ 
    TiXmlElement* secondChild = child->FirstChild("featureVector")->ToElement();
    vector<FeatureVector> sequence;
    vector<FeatureVector> vals;
    for(secondChild; secondChild ; secondChild = secondChild->NextSiblingElement("featureVector")){
      int length = 0;
      if(TIXML_SUCCESS!=secondChild->QueryIntAttribute("length",&length)){
	throw invalid_argument("RegressionDataset : Uncorrect attribute name");
      }
      FeatureVector fv(length);
      stringstream ss (stringstream::in | stringstream::out);
      ss <<  secondChild->GetText();
      for (int n=0; n<length; n++){
	ss >> fv[n];
      }
      sequence.push_back(fv);
    }
    secondChild = child->FirstChild("values")->ToElement();
    for(secondChild; secondChild ; secondChild = secondChild->NextSiblingElement("values")){
      int length = 0;
      if(TIXML_SUCCESS!=secondChild->QueryIntAttribute("length",&length)){
	throw invalid_argument("RegressionDataset : Uncorrect attribute name");
      }
      FeatureVector fv(length);
      stringstream ss (stringstream::in | stringstream::out);
      ss <<  secondChild->GetText();
      for (int n=0; n<length; n++){
	ss >> fv[n];
      }
      vals.push_back(fv);
    }
    addSequence(sequence,vals);
  }
}
  
void RegressionDataset::save(std::string _fileName){
  TiXmlDocument doc;
  TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "no" );
  doc.LinkEndChild( decl );
  TiXmlElement * classificationDataset = new TiXmlElement( "regressionDataset" );
  TiXmlElement * datasetInfos = new TiXmlElement( "datasetInfos" );
  TiXmlElement * dname = new TiXmlElement( "name" );
  TiXmlText * tname = new TiXmlText( name );
  dname->LinkEndChild(tname);
  datasetInfos->LinkEndChild(dname);

  classificationDataset->LinkEndChild( datasetInfos );
  TiXmlElement * ddata = new TiXmlElement( "data" );
  for(uint i=0;i<data.size();i++){
    TiXmlElement * sequence = new TiXmlElement( "sequence" );
    for(uint j=0;j<data[i].size();j++){
      TiXmlElement * fv = new TiXmlElement( "featureVector" ); 
      fv->SetAttribute("length",data[i][j].getLength());
      stringstream ss(stringstream::in | stringstream::out);
      for(uint k=0;k<data[i][j].getLength();k++){
	ss << data[i][j][k] << " ";
      }
      TiXmlText * tfv = new TiXmlText( ss.str());
      fv->LinkEndChild(tfv);
      sequence->LinkEndChild(fv);
    }
    for(uint j=0;j<data[i].size();j++){
      TiXmlElement * values = new TiXmlElement( "values" ); 
      values->SetAttribute("length",data[i][j].getLength());
      stringstream ss(stringstream::in | stringstream::out);
      for(uint k=0;k<getTargetSample(i,j).getLength();k++){
	ss << getTargetSample(i,j)[k] << " ";
      }
      TiXmlText * tfv = new TiXmlText( ss.str());
      values->LinkEndChild(tfv);
      sequence->LinkEndChild(values);
    }
    ddata->LinkEndChild(sequence);
  }
  classificationDataset->LinkEndChild(ddata);
  doc.LinkEndChild( classificationDataset );
  doc.SaveFile( _fileName );
}

vector<FeatureVector> RegressionDataset::getTargetSequence(uint _i) const{
  return values[_i];
}

FeatureVector RegressionDataset::getTargetSample(uint _i, uint _j) const{
  return values[_i][_j];
}

RegressionDataset::~RegressionDataset(){

}

ostream& operator<<(ostream& _os, RegressionDataset& _rd){
  _os << "Regression dataset '" << _rd.getName() << "' : " << endl;
  _os << "\t - Sequences : " << _rd.getNumSequences() << endl;
  _os << "\t - Samples : " << _rd.getNumSamples() << endl;
  for(uint i=0;i<_rd.getNumSequences();i++){
    _os << "Sequence "<< i <<"[ "<<endl;
    _os << "Features (" << endl;
    for(uint j=0;j<_rd[i].size();j++){
      _os << _rd[i][j] ;
    }
    _os << " ) " << endl << "Values (" << endl;
    vector<FeatureVector> target = _rd.getTargetSequence(i);
    for(uint j=0;j<target.size();j++){
      _os << target[j] ;
    }
    _os << " )] ";
  }
  return _os;
}
