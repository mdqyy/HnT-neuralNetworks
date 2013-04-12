/*!
 * \file ImageProcessing.cpp
 * Body of the Image Processing class.
 * \author Luc Mioulet
 */
#include "ImageProcessing.hpp"

using namespace std;
using namespace cv;

FeatureVector extractBlackAndWhiteFrame(Mat _image, int _horizontalStartingPoint, uint _frameLength) {
  FeatureVector fv(_frameLength * _image.rows);
  for (uint j = 0; j < _frameLength; j++) {
    for (int k = 0; k < _image.rows; k++) {
      if ((int) _image.at<uchar>(k, _horizontalStartingPoint + j) > 0) {
	fv[j * _image.rows + k] = 1.0;
      } else {
	fv[j * _image.rows + k] = 0.0;
      }
    }
  }
  return fv;
}

FeatureVector extractBlackAndWhiteFrame(Mat _image, int _horizontalStartingPoint, uint _frameLength, pair<int,int> _frameZone) {
  FeatureVector fv(_frameLength * (_frameZone.second - _frameZone.first));
  int i=0;
  for (uint j = 0; j < _frameLength; j++) {
    for (int k = _frameZone.first; k < _frameZone.second; k++) {
      if ((int) _image.at<uchar>(k, _horizontalStartingPoint + j) > 0) {
	fv[i] = 1.0;
      } else {
	fv[i] = 0.0;
      }
      i++;
    }
  }
  return fv;
}

vector<FeatureVector> extractFrames(Mat _image, int _frameLength) {
  int subparts = floor(((float) _image.cols) / ((float) _frameLength));
  vector<FeatureVector> features = vector<FeatureVector>();
  for (int i = 0; i < subparts; i++) {
    features.push_back(extractBlackAndWhiteFrame(_image, i * _frameLength, _frameLength));
  }
  return features;
}

vector<FeatureVector> extractOverlappingFrames(Mat _image, int _frameLength) {
  int subparts = floor(((float) _image.cols) / ((float) _frameLength));
  subparts = subparts*2 -1 ; // length is doubled minus one
  vector<FeatureVector> features = vector<FeatureVector>();
  for (int i = 0; i < subparts; i++) {
    if (i % 2 == 0){
      features.push_back(extractBlackAndWhiteFrame(_image, i * _frameLength, _frameLength));
    }
    else{
      features.push_back(extractBlackAndWhiteFrame(_image, i * _frameLength-_frameLength/2, _frameLength)); //start at half the length
    }
  }
  return features;
}

vector<FeatureVector> extractOverlappingFramesPPerP(Mat _image, int _frameLength) {
  vector<FeatureVector> features = vector<FeatureVector>();
  for (int i = 0; i < _image.cols - _frameLength; i++) {
      features.push_back(extractBlackAndWhiteFrame(_image, i, _frameLength));
  }
  return features;
}


vector<FeatureVector> extractFrames(Mat _image, int _frameLength, pair<int,int> _frameZone) {
  int subparts = floor(((float) _image.cols) / ((float) _frameLength));
  vector<FeatureVector> features = vector<FeatureVector>();
  for (int i = 0; i < subparts; i++) {
    features.push_back(extractBlackAndWhiteFrame(_image, i* _frameLength, _frameLength, _frameZone));
  }
  return features;
}

Mat buildImage(vector<FeatureVector> _sequence, int _frameLength) {
  int frameHeight = _sequence[0].getLength() / _frameLength;
  int imageLength = _frameLength * _sequence.size();
  Mat image = Mat(frameHeight, imageLength, CV_8UC1, Scalar(0));
  int frameIndex = 0;
  uint sequenceIndex = 0;
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      image.at<uchar>(j, i) = _sequence[frameIndex][sequenceIndex] * 255;
      sequenceIndex += 1;
      if (sequenceIndex >= _sequence[0].getLength()) {
	sequenceIndex = 0;
	frameIndex += 1;
      }
    }
  }
  return image;
}

Mat buildFrame(FeatureVector _fv, int _frameLength) {
  int frameHeight = _fv.getLength() / _frameLength;
  Mat image = Mat(frameHeight, _frameLength, CV_8UC1, Scalar(0));
  uint sequenceIndex = 0;
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      image.at<uchar>(j, i) = _fv[sequenceIndex] * 255;
      sequenceIndex += 1;
    }
  }
  return image;
}

Mat buildColorMapImage(vector<FeatureVector> _sequence, int _frameLength, vector<int> _colorSequence, vector<Vec3b> _colorMap) {
  int frameHeight = _sequence[0].getLength() / _frameLength;
  int imageLength = _frameLength * _sequence.size();
  Mat image = Mat(frameHeight, imageLength, CV_8UC3, Scalar(0));
  int frameIndex = 0;
  uint sequenceIndex = 0;
  int colorValue = 0;
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      for(int k=0;k<3;k++){
	colorValue = ((int)_colorMap[_colorSequence[frameIndex]].val[k])*(1.0 - _sequence[frameIndex][sequenceIndex]);
	image.at<Vec3b>(j, i)[k] = ((uchar)colorValue);
      }
      sequenceIndex += 1;
      if (sequenceIndex >= _sequence[0].getLength()) {
	sequenceIndex = 0;
	frameIndex += 1;
      }
    }
  }
  return image;
}

Mat buildColorFrame(FeatureVector _fv, int _frameLength, Vec3b _color) {
  int frameHeight = _fv.getLength() / _frameLength;
  Mat image = Mat(frameHeight, _frameLength, CV_8UC3);
  uint sequenceIndex = 0;
  int colorValue = 0;
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      for(int k=0;k<2;k++){
	colorValue = ((int)_color.val[k])*(1.0 - _fv[sequenceIndex]);
	image.at<Vec3b>(j, i)[k] = ((uchar)colorValue);
      }
      sequenceIndex += 1;
    }
  }
  return image;
}

vector<Vec3b> createColorRepartition(uint _numColors, uint _saturation, uint _value){
  Mat map = Mat(_numColors,1,CV_8UC3);
  cvtColor(map,map,CV_BGR2HSV);
  vector<Vec3b> colors;
  for(uint i = 0 ; i< _numColors; i++){
    map.at<Vec3b>(i,0)[0] = ((uchar)((180.0 / _numColors)*i));
    map.at<Vec3b>(i,0)[1] = ((uchar)(_saturation));
    map.at<Vec3b>(i,0)[2] = ((uchar)(_value));
  }
  cvtColor(map,map,CV_HSV2BGR);
  for(uint i = 0 ; i< _numColors; i++){
    colors.push_back(map.at<Vec3b>(i,0));
  }
  return colors;
}
