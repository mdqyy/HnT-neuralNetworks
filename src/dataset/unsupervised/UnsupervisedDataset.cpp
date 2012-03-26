/*!
 * \file UnsupervisedDataset.cpp
 * Body of the UnsupervisedDataset class.
 * \author Luc Mioulet
 */

#include "UnsupervisedDataset.hpp"

using namespace std;
using namespace cv;

UnsupervisedDataset::UnsupervisedDataset(): Dataset(){

}

int UnsupervisedDataset::getDatasetType() const{
  return DS_UNSUPERVISED;
}

void UnsupervisedDataset::addSequence(vector<FeatureVector> sequence){
  data.push_back(sequence);
  if( sequence.size() > maxSequenceLength){
    maxSequenceLength = sequence.size();
  }
  updateStatistics(sequence);
}

void UnsupervisedDataset::addSample(FeatureVector sample, int index){
  int insertIndex=index;
  if(insertIndex<0){
    data.push_back(vector<FeatureVector>(1,sample));
  }
  else{
    data[index].push_back(sample);
    if( data[index].size() > maxSequenceLength){ //check if we de not go over the actual maximum 
      maxSequenceLength = data[index].size(); 
    }
  }
  updateStatistics(sample);
}

void UnsupervisedDataset::load(std::string fileName){
  /* Open file */
  TiXmlDocument doc( fileName );
  if ( !doc.LoadFile() ){
    throw invalid_argument("UnsupervisedDataset : Uncorrect filename");
  }
  TiXmlHandle hdl(&doc);
  /* Get Dataset infos */
  TiXmlElement *elem = hdl.FirstChildElement("unsupervisedDataset").FirstChildElement("datasetInfos").FirstChildElement("name").Element();
  name = elem->GetText();
  /* Fill in the feature vectors */
  TiXmlElement* child = hdl.FirstChild( "unsupervisedDataset" ).FirstChild( "data" ).FirstChild( "sequence" ).ToElement();  
  for(child; child ;child = child->NextSiblingElement()){ 
    TiXmlElement* secondChild = child->FirstChild("featureVector")->ToElement();
    vector<FeatureVector> sequence;
    vector<FeatureVector> vals;
    for(secondChild; secondChild ; secondChild = secondChild->NextSiblingElement("featureVector")){
      int length = 0;
      if(TIXML_SUCCESS!=secondChild->QueryIntAttribute("length",&length)){
	throw invalid_argument("UnsupervisedDataset : Uncorrect attribute name");
      }
      FeatureVector fv(length);
      stringstream ss (stringstream::in | stringstream::out);
      ss <<  secondChild->GetText();
      for (int n=0; n<length; n++){
	ss >> fv[n];
      }
      sequence.push_back(fv);
    }
    addSequence(sequence);
  }
}
  
void UnsupervisedDataset::save(std::string fileName){
  TiXmlDocument doc;
  TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "no" );
  doc.LinkEndChild( decl );
  TiXmlElement * classificationDataset = new TiXmlElement( "unsupervisedDataset" );
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
    ddata->LinkEndChild(sequence);
  }
  classificationDataset->LinkEndChild(ddata);
  doc.LinkEndChild( classificationDataset );
  doc.SaveFile( fileName );
}

UnsupervisedDataset::~UnsupervisedDataset(){

}

ostream& operator<<(ostream& os, UnsupervisedDataset& rd){
  os << "Unsupervised dataset '" << rd.getName() << "' : " << endl;
  os << "\t - Sequences : " << rd.getNumSequences() << endl;
  os << "\t - Samples : " << rd.getNumSamples() << endl;
  for(uint i=0;i<rd.getNumSequences();i++){
    os << "Sequence "<< i <<"[ "<<endl;
    os << "Features (" << endl;
    for(uint j=0;j<rd[i].size();j++){
      os << rd[i][j] ;
    }
  }
  return os;
}
