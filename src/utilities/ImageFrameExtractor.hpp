#ifndef __IMAGEFRAMEEXTRACTOR_HPP__
#define __IMAGEFRAMEEXTRACTOR_HPP__
/*!
 * \file ImageFrameExtractor.hpp
 * Header of the ImageFrameExtractor class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include "../dataset/FeatureVector.hpp"

/*!
 * \class ImageFrameExtractor
 * Description
 */
class ImageFrameExtractor {
 private :

 protected:
  /* Image scalinge */
  realv scale;
  /* Frame size */
  uint frameSize;
  /* Space between two frames */
  uint interFrameSpace;

 public:

  /*!
   * Default constructor.
   */
  ImageFrameExtractor();

  /*!
   * Parameter constructor.
   */
  ImageFrameExtractor(realv _scale, uint _frameSize, uint _interFrameSpace);

  /*!
   * Get the scale
   * \return The scale used on the images.
   */
  realv getScale();

  /*!
   * Get the frame size.
   * \return The frame size.
   */
  uint getFrameSize();

  /*!
   * Get the space between the extraction of two frames.
   * \return The space between two frames.
   */
  uint getInterFrameSpace();
  
  /*!
   * Get the frame i at a particuliar position on the base image.
   * \param _image The input image.
   * \param _frame The original central colon.
   * \return The feature vector of this frame.
   */
  FeatureVector getFrameCenteredOn(cv::Mat _image,uint _row);

  /*!
   * Get the frame i in an image.
   * \param _image The input image.
   * \param _frame The number of the frame.
   * \return The feature vector of this frame.
   * \warning The frame must exist in the image !!
   */
  FeatureVector getOneFrame(cv::Mat _image,uint _frame);

  /*!
   * Get the frames from an image.
   * \param _image The input image.
   * \return The vector of feature vectors of this image.
   */
  std::vector<FeatureVector> getFrames(cv::Mat _image);
  
  /*!
   * Destructor.
   */
  ~ImageFrameExtractor();

};


#endif
