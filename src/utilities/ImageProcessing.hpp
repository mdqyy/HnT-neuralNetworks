#ifndef IMAGEPROCESSING_HPP_
#define IMAGEPROCESSING_HPP_

/*!
 * \file ImageProcessing.hpp
 * Header of the Image Processing class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
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
 * Extract a frame from an image.
 * \param _image The image.
 * \param _horizontalStartingPoint Starting horizontal point.
 * \param _frameLength Frame Length.
 * \param _frameZone A given frame zone to extract.
 * \return Feature vector of the frame.
 */
FeatureVector extractBlackAndWhiteFrame(cv::Mat _image, int _horizontalStartingPoint,uint _frameLength, std::pair<int,int> _frameZone);

/*!
 * Extract frames of an images.
 * \param _image The image.
 * \param _frameLength Frame Length.
 * \return The feature vector of frames.
 */
std::vector<FeatureVector> extractFrames(cv::Mat _image, int _frameLength);

/*!
 * Extract frames of an images.
 * \param _image The image.
 * \param _frameLength Frame Length.
 * \param _frameZone A given frame zone to extract.
 * \return The feature vector of frames.
 */
std::vector<FeatureVector> extractFrames(cv::Mat _image, int _frameLength, std::pair<int,int> _frameZone);

/*!
 * Rebuild an image from a sequence.
 * \param _sequence The sequence.
 * \param _frameLength The frame length used so build the sequence.
 * \return The sequence image.
 */
cv::Mat buildImage(std::vector<FeatureVector> _sequence,int _frameLength);

/*!
 * Rebuild an image from a sequence adding colors corresponding.
 * \param _sequence The sequence.
 * \param _frameLength The frame length used so build the sequence.
 * \param _colorSequence Color attribution of the sequence.
 * \param _colorMap Color map for the sequence.
 * \return The sequence image.
 */
cv::Mat buildColorMapImage(std::vector<FeatureVector> _sequence, int _frameLength, std::vector<int> _colorSequence, std::vector<cv::Vec3b> _colorMap);

/*!
 * Build a frame from a feature vector.
 * \param _fv The feature vector.
 * \param _frameLength The frame length.
 * \return The frame image.
 */
cv::Mat buildFrame(FeatureVector _fv, int _frameLength);

/*!
 * Build a color frame from a feature vector.
 * \param _fv The feature vector.
 * \param _frameLength The frame length.
 * \param _color The color.
 * \return The frame image.
 */
cv::Mat buildColorFrame(FeatureVector _fv, int _frameLength, cv::Vec3b _color);

/*!
 * Build a color vector of unique colors.
 * \param _numColors Number of different colors.
 * \param _saturation Saturation value in HSV color space.
 * \param _value Value in HSV color space.
 * \return A vector of colors.
 */
std::vector<cv::Vec3b> createColorRepartition(uint _numColors, uint _saturation = 200, uint _value = 200);
#endif /* IMAGEPROCESSING_HPP_ */
