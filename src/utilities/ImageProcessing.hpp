#ifndef IMAGEPROCESSING_HPP_
#define IMAGEPROCESSING_HPP_

/*!
 * \file ImageProcessing.hpp
 * Header of the Image Processing class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <vector>
#include "../dataset/FeatureVector.hpp"

/*!
 * Extract a frame from an image.
 * \param _image The image.
 * \param _horizontalStartingPoint Starting horizontal point.
 * \param _frameLength Frame Length.
 * \return Feature vector of the frame.
 */
FeatureVector extractBlackAndWhiteFrame(cv::Mat _image, int _horizontalStartingPoint,uint _frameLength);

/*!
 * Extract frames of an images.
 * \param _image The image.
 * \param _frameLength Frame Length.
 * \return The feature vector of frames.
 */
std::vector<FeatureVector> extractFrames(cv::Mat _image, int _frameLength);

/*!
 * Rebuild an image from a sequence.
 * \param _sequence The sequence.
 * \param _frameLength The frame length used so build the sequence.
 * \return The sequence image.
 */
cv::Mat buildImage(std::vector<FeatureVector> _sequence,int _frameLength);

/*!
 * Build a frame from a feature vector.
 * \param _fv The feature vector.
 * \param _frameLength The frame length.
 * \return The frame image.
 */
cv::Mat buildFrame(FeatureVector _fv, int _frameLength);
#endif /* IMAGEPROCESSING_HPP_ */
