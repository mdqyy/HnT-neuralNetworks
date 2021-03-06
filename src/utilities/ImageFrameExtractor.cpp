/*!
 * \file ImageFrameExtractor.cpp
 * Body of the ImageFrameExtractor class.
 * \author Luc Mioulet
 */

#include "ImageFrameExtractor.hpp"

using namespace std;
using namespace cv;

ImageFrameExtractor::ImageFrameExtractor() :scale(1),frameSize(1), interFrameSpace(1) {

}

ImageFrameExtractor::ImageFrameExtractor(realv _scale, uint _frameSize, uint _interFrameSpace) : scale(_scale),frameSize(_frameSize), interFrameSpace(_interFrameSpace){

}

realv ImageFrameExtractor::getScale() const{
  return scale;
}

void ImageFrameExtractor::setScale(realv _scale){
  scale = _scale;
}

uint ImageFrameExtractor::getFrameSize() const{
  return frameSize;
}

void ImageFrameExtractor::setFrameSize(uint _frameSize){
  frameSize = _frameSize;
}

uint ImageFrameExtractor::getInterFrameSpace() const{
  return interFrameSpace;
}

void ImageFrameExtractor::setInterFrameSpace(uint _interFrameSpace){
  interFrameSpace = _interFrameSpace;
}

FeatureVector ImageFrameExtractor::getFrameCenteredOn(Mat _image,uint _row) const{
  Mat imageProc;
  resize(_image,imageProc,Size(0,0),scale,scale,INTER_LINEAR);
  uint fvLength = imageProc.rows*frameSize;
  FeatureVector result(fvLength);
  int start = _row*scale-frameSize/2;
  int stop = start + frameSize;
  uint index = 0;
  for(int col = start;col<stop && col<imageProc.cols-1;col++){
    for(int row = 0; row<imageProc.rows;row++){
      if(col >=0){
	if((int)imageProc.at<uchar>(row,col) > 120){/*Bad hard coded threshold ! */
	  result[index]=1.0;
	}
	else{
	  result[index]=0.0;
	}
      }
      index++;
    }
  }
  return result;
}

FeatureVector ImageFrameExtractor::getOneFrame(Mat _image,uint _frame) const{
  Mat imageProc;
  resize(_image,imageProc,Size(0,0),scale,scale,INTER_LINEAR);
  uint numberOfFrames = imageProc.cols/interFrameSpace;
  uint fvLength = imageProc.rows*frameSize;
  FeatureVector result(fvLength);
  uint index = 0;
  if(imageProc.cols%interFrameSpace!=0){ 
    numberOfFrames ++;
  }
  if(_frame>numberOfFrames){
    cerr << "The frame you want does not exist" << endl;
  }
  for(uint col = _frame*interFrameSpace;col<_frame*interFrameSpace+frameSize && col<imageProc.cols-1;col++){
    for(uint row = 0; row<imageProc.rows;row++){
      if((int)imageProc.at<uchar>(row,col) > 120){/*Bad hard coded threshold ! */
	result[index]=1.0;
      }
      else{
	result[index]=0.0;
      }
      index++;
    }
  }
  return result;
}

vector<FeatureVector> ImageFrameExtractor::getFrames(Mat _image){
  vector<FeatureVector> frames = vector<FeatureVector>();
  Mat imageProc;
  bool residue = false;
  resize(_image,imageProc,Size(0,0),scale,scale,INTER_LINEAR);
  uint numberOfFrames = imageProc.cols/interFrameSpace;
  uint fvLength = imageProc.rows*frameSize;
  if(imageProc.cols%interFrameSpace!=0){ 
    numberOfFrames ++;
  }
  for(uint i=0;i<numberOfFrames;i++){
    FeatureVector vec(fvLength);
    int index = 0;
    for(uint col = i*interFrameSpace;col<i*interFrameSpace+frameSize && col<imageProc.cols-1;col++){
      for(uint row = 0; row<imageProc.rows;row++){
	if((int)imageProc.at<uchar>(row,col) > 120){/*Bad hard coded threshold ! */
	  vec[index]=1.0;
	}
	else{
	  vec[index]=0.0;
	}
	index++;
      }
    }
    frames.push_back(vec);
  }
  return frames;
}

ImageFrameExtractor::~ImageFrameExtractor(){

}


ofstream& operator<<(ofstream& _ofs, const ImageFrameExtractor& _ife){
  _ofs << " < ";
  _ofs << _ife.getScale() <<" " << _ife.getFrameSize() << " " << _ife.getInterFrameSpace()<< endl;
  _ofs << " > ";
  return _ofs;
}

ifstream& operator>>(ifstream& _ifs, ImageFrameExtractor& _ife){
  string temp;
  uint uintValue;
  realv realValue;
  _ifs >> temp;
  _ifs >> realValue;
  _ife.setScale(realValue);
  _ifs >> uintValue ;
  _ife.setFrameSize(uintValue);
  _ifs >> uintValue;
  _ife.setInterFrameSpace(uintValue);
  _ifs >> temp;
  return _ifs;
}
