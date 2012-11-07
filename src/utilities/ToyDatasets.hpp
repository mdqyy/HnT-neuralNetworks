#ifndef TOYDATASETS_HPP_
#define TOYDATASETS_HPP_

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <stdio.h>
#include <fstream>
#include <sstream>

#include "../dataset/supervised/RegressionDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/FeatureVector.hpp"

/*!
 * \file ToyDatasets.hpp
 * Header of the Toy Datasets class.
 * \author Luc Mioulet
 */

/*!
 * Draws a gaussian distributed x-y dataset.
 * \param _dataset Pointer to the dataset.
 * \param _numberOfPoints Number of points in the dataset.
 * \param _centerX Center position in X.
 * \param _centerY Center position in Y.
 * \param _standardDeviationX Standard deviation in X.
 * \param _standardDeviationY Standard deviation in Y.
 */
void circlePotato(RegressionDataset *_dataset, int _numberOfPoints, realv _centerX, realv _centerY, realv _standardDeviationX, realv _standardDeviationY);

/*!
 * Draws a smile-like x-y dataset.
 * \param _dataset Pointer to the dataset.
 * \param _numberOfPoints Number of points in the dataset.
 * \param _centerX Center position in X.
 * \param _centerY Center position in Y.
 * \param _standardDeviationX Standard deviation in X.
 * \param _standardDeviationY Standard deviation in Y.
 */
void smilePotato(RegressionDataset *_dataset, int _numberOfPoints, realv _centerX, realv _centerY, realv _standardDeviationX, realv _standardDeviationY);


/*!
 * Draws an inversed smile-like x-y dataset.
 * \param _dataset Pointer to the dataset.
 * \param _numberOfPoints Number of points in the dataset.
 * \param _centerX Center position in X.
 * \param _centerY Center position in Y.
 * \param _standardDeviationX Standard deviation in X.
 * \param _standardDeviationY Standard deviation in Y.
 */
void inversedSmilePotato(RegressionDataset *_dataset, int _numberOfPoints, realv _centerX, realv _centerY, realv _standardDeviationX, realv _standardDeviationY);

#endif /* TOYDATASETS_HPP_ */
