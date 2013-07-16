#ifndef __IMAGEDATASET_HPP__
#define __IMAGEDATASET_HPP__
/*!
 * \file ImageDataset.hpp
 * Header of the RegressionDataset class.
 * \author Luc Mioulet
 */

#include <string>
#include <vector>
#include <iostream>
#include <opencv/cv.h>
#include "../utilities/ImageFrameExtractor.hpp"

/*!
 * \class ImageDataset
 * Extract image
 */
class ImageDataset {
 private :
  std::vector< std::string > images;
  ImageFrameExtractor ife;
 protected:

 public:

  /*!
   * Default constructor.
   */
  ImageDataset();

  /*!
   * Parameter constructor.
   * \param _ife Image Frame Extractor.
   * \param _images Vector of strings.
   */
  ImageDataset(std::vector< std::string > _images, ImageFrameExtractor _ife=ImageFrameExtractor(1.0,1,1));

  /*! Add an image.
   * \param _image image name.
   */
  void addImage(std::string _image);
  
  /*! Get the Image Frame Extractor
   * \return Image frame extractor.
   */
  ImageFrameExtractor getImageFrameExtractor();

  /*! Set the Image Frame Extractor.
   * \param _ife Image frame extractor.
   */
  void setImageFrameExtractor(ImageFrameExtractor _ife);

  /* Get number of images.
   * \return Number of images.
   */
  int getNumberOfImages();

  /* Get images.
   * \return The list of images.
   */
  std::vector< std::string > getImages();

  /*Get image at a certain index.
   * \param _i Index position.
   * \warn If out of bound, will output an error.
   * \return Image at the index.
   */
  std::string getImage(uint _i);

  /*Get the matrix image at a certain index.
   * \param _i Index position.
   * \param _mode Opening mode (OPENCV option).
   * \warn If out of bound, will output an error.
   * \return Matrix of the image at index.
   */
  cv::Mat getMatrix(uint _i,int _mode=0);

  /*Get the features of image at a certain index.
   * \param _i Index position.
   * \warn If out of bound, will output an error.
   * \return vector of feature vectors.
   */
  std::vector< FeatureVector > getFeatures(uint _i);
  
  /*!
   * Load a database from a file.
   * \param _fileName
   */
  void load(std::string _fileName);

  /*!
   * Save a database to a file.
   * \param _fileName
   */
  void save(std::string _fileName);

  /*!
   * Destructor.
   */
  ~ImageDataset();

  /*!
   * Output stream for the dataset.
   * \param _os Output stream.
   * \param _rd Regression dataset.
   * \return Output stream.
   */
  friend  std::ostream& operator<<(std::ostream& _os, ImageDataset& _id);
};


#endif
