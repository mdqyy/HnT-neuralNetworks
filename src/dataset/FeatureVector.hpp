#ifndef __FEATUREVECTOR_HPP__
#define __FEATUREVECTOR_HPP__
/*!
 * \file FeatureVector.hpp
 * Header of the FeatureVector class.
 * \author Luc Mioulet
 */

#include "ValueVector.hpp"
#include <opencv/cv.h>


/*!
 * \class FeatureVector
 * Contains the feature vector given to Machines.
 */
class FeatureVector : public ValueVector{
 private :

 protected:

 public:


  /*!
   * Parameter constructor.
   * \param _length Vector length.
   */
  FeatureVector(int _length=1);
  

  /*!
   * Parameter constructor.
   * \param _data Data of the feature vector.
   */
  FeatureVector(cv::Mat _data);

  /*!
   * Destructor.
   */
  ~FeatureVector();

  /*!
   * Output feature vector data.
   * \param os Output stream.
   * \param fv Feature vector.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const FeatureVector& fv);

};


#endif
