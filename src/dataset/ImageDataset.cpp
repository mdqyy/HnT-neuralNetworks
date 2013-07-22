/*!
 * \file ImageDataset.cpp
 * Body of the ImageDataset class.
 * \author Luc Mioulet
 */

#include "ImageDataset.hpp"

using namespace std;
using namespace cv;

ImageDataset::ImageDataset() :images(vector<string>()), ife(ImageFrameExtractor(1.0,1,1)) {
}

ImageDataset::ImageDataset(vector< string > _images,ImageFrameExtractor _ife) : images(_images) ,  ife(_ife){
}


void ImageDataset::addImage(string _image){
  Mat temp = imread(_image,0);
  if(!temp.empty()){
    images.push_back(_image);
  }
  else{
    cout << "Image " << _image << "doesn't exist or is corrupted, and was not added" << endl;
  }
}
 
ImageFrameExtractor ImageDataset::getImageFrameExtractor(){
  return ife;
}


void ImageDataset::setImageFrameExtractor(ImageFrameExtractor _ife){
  ife = _ife;
}


int ImageDataset::getNumberOfImages(){
  return images.size();
}

vector< string > ImageDataset::getImages(){
  return images;
}

string ImageDataset::getImage(uint _i){
  return images[_i];
}


Mat ImageDataset::getMatrix(uint _i, int _mode){
  return imread(images[_i],_mode);
}
 
vector< FeatureVector > ImageDataset::getFeatures(uint _i){
  return ife.getFrames(imread(images[_i],0));
}

void ImageDataset::load(string _fileName){
  ifstream ifs;
  ifs.open(_fileName.c_str());
  string file;
  while(!ifs.eof()){
    ifs >> file;
    addImage(file);
  }
}


void ImageDataset::save(string _fileName){
  ofstream ofs;
  ofs.open(_fileName.c_str());
  for(uint i=0;i<images.size();i++){
    ofs << images[i] << endl;
  }
}


ImageDataset::~ImageDataset(){

}


ostream& operator<<(ostream& _os, ImageDataset& _id){
    _os << "Image dataset containing " <<_id.getNumberOfImages() << " images" << endl;
    return _os;
}
