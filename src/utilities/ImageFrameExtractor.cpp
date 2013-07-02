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

realv ImageFrameExtractor::getScale(){
  return scale;
}

uint ImageFrameExtractor::getFrameSize(){
  return frameSize;
}

uint ImageFrameExtractor::getInterFrameSpace(){
  return interFrameSpace;
}

FeatureVector ImageFrameExtractor::getOneFrame(Mat _image,uint _frame){
  Mat imageProc;
  bool residue = false;
  resize(_image,imageProc,Size(0,0),scale,scale,INTER_LINEAR);
  uint numberOfFrames = imageProc.cols/interFrameSpace;
  uint fvLength = imageProc.rows*frameSize;
  FeatureVector result(fvLength);
  if(imageProc.cols%interFrameSpace!=0){ 
    numberOfFrames ++;
  }
  if(_frame>numberOfFrames){
    cerr << "The frame you want does not exist" << endl;
  }
  for(uint col = _frame*interFrameSpace;col<_frame*interFrameSpace+frameSize && col<imageProc.cols-1;col++){
    for(uint row = 0; row<imageProc.rows;row++){
      if((int)imageProc.at<uchar>(row,col) > 120){/*Bad hard coded threshold ! */
	fvLength[index]=1;
      }
      else{
	fvLength[index]=0;
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
	  vec[index]=1;
	}
	else{
	  vec[index]=0;
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
