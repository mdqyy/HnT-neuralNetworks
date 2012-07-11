/*!
 * \file ImageProcessing.cpp
 * Body of the Image Processing class.
 * \author Luc Mioulet
 */
#include "ImageProcessing.hpp"

using namespace std;
using namespace cv;

FeatureVector extractBlackAndWhiteFrame(Mat _image, int _horizontalStartingPoint,uint _frameLength){
	FeatureVector fv(_frameLength*_image.rows);
	for(uint j=0;j<_frameLength;j++){
		for(int k=0;k<_image.rows;k++){
			if((int)_image.at<uchar>(k,_horizontalStartingPoint*_frameLength+j)>0){
				fv[j*_image.rows+k]=1.0;
			}
			else{
				fv[j*_image.rows+k]=0.0;
			}
		}
	}
	return fv;
}

vector<FeatureVector> extractFrames(Mat _image, int _frameLength){
	int subparts=floor(((float)_image.cols)/((float)_frameLength));
	cout << subparts << endl;
	vector<FeatureVector> features = vector<FeatureVector>();
	for(int i=0;i<subparts;i++){
		features.push_back(extractBlackAndWhiteFrame(_image,i,_frameLength));
	}
	return features;
}

Mat buildImage(vector<FeatureVector> _sequence,int _frameLength){
	int frameHeight = _sequence[0].getLength()/_frameLength;
	int imageLength = _frameLength*_sequence.size();
	Mat image = Mat(frameHeight,imageLength,CV_8UC1,Scalar(0));
	int frameIndex = 0;
	uint sequenceIndex = 0;
	for(int i=0;i<image.cols;i++){
		for(int j=0;j<image.rows;j++){
			image.at<uchar>(j,i)=_sequence[frameIndex][sequenceIndex]*255;
			sequenceIndex += 1;
			cout << image.at<uchar>(j,i) << endl;
			if(sequenceIndex >= _sequence[0].getLength()){
						sequenceIndex = 0;
						frameIndex += 1;
			}
		}
	}
	return image;
}

Mat buildFrame(FeatureVector _fv, int _frameLength){
	int frameHeight = _fv.getLength()/_frameLength;
	Mat image = Mat(frameHeight, _frameLength,CV_8UC1,Scalar(0));
	uint sequenceIndex = 0;
	for(int i=0;i<image.cols;i++){
		for(int j=0;j<image.rows;j++){
			image.at<uchar>(j,i)=_fv[sequenceIndex]*255;
			sequenceIndex += 1;
		}
	}
	return image;
}
