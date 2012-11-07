/*!
 * \file ImageProcessing.cpp
 * Body of the Image Processing class.
 * \author Luc Mioulet
 */
#include "ToyDatasets.hpp"

using namespace std;
using namespace cv;

void circlePotato(RegressionDataset *_dataset, int _numberOfPoints, realv _centerX, realv _centerY, realv _standardDeviationX, realv _standardDeviationY) {
	RNG random(getTickCount());
	random.next();
	for (int i = 0; i < _numberOfPoints; i++) {
		vector<FeatureVector> point;
		FeatureVector fv(2);
		fv[0] = random.gaussian(_standardDeviationX) + _centerX;
		fv[1] = random.gaussian(_standardDeviationY) + _centerY;
		point.push_back(fv);
		_dataset->addSequence(point,point);
	}
}

void smilePotato(RegressionDataset *_dataset, int _numberOfPoints, realv _centerX, realv _centerY, realv _standardDeviationX, realv _standardDeviationY) {
	RNG random(getTickCount());
	random.next();
	for (int i = 0; i < _numberOfPoints; i++) {
		vector<FeatureVector> point;
		FeatureVector fv(2);
		fv[0] = random.gaussian(_standardDeviationX);
		fv[1] = (fv[0] * fv[0]) + random.gaussian(_standardDeviationY) + _centerY;
		fv[0] += _centerX;
		point.push_back(fv);
		_dataset->addSequence(point,point);
	}
}

void inversedSmilePotato(RegressionDataset *_dataset, int _numberOfPoints, realv _centerX, realv _centerY, realv _standardDeviationX, realv _standardDeviationY) {
	RNG random(getTickCount());
	random.next();
	for (int i = 0; i < _numberOfPoints; i++) {
		vector<FeatureVector> point;
		FeatureVector fv(2);
		fv[0] = random.gaussian(_standardDeviationX);
		fv[1] = -(fv[0] * fv[0]) + random.gaussian(_standardDeviationY) + _centerY;
		fv[0] += _centerX;
		point.push_back(fv);
		_dataset->addSequence(point,point);
	}
}
