#ifndef __CLASSIFICATIONERRORMEASURER_HPP__
#define __CLASSIFICATIONERRORMEASURER_HPP__
/*!
 * \file ClassificationErrorMeasurer.hpp
 * Header of the ClassificationErrorMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <opencv/cv.h>



/*!
 * \class ClassificationErrorMeasurer
 * Measure classification errors between a machine output vector and the data target.
 */
class ClassificationErrorMeasurer : ErrorMeasurer{
 private :
  uint numberOfExamples;

 protected:

 public:

  /*!
   * Default constructor.
   */
  ClassificationErrorMeasurer(uint _numberOfExamples);
  
  /*!
   * Measure error per unit.
   * \param ouput Machine output feature Vector.
   * \param target Dataset result to match.
   * \return The error vector. Has the same length as the output/target.
   */
  virtual ErrorVector errorPerUnit(FeatureVector _output, FeatureVector _target);

  /*!
   * Measure error per unit.
   * \param ouput Machine output feature Vector.
   * \param target Dataset result to match.
   * \return The total error.
   */
  virtual realv totalError(FeatureVector _output, FeatureVector _target);

  /*!
   * Destructor.
   */
  ~ClassificationErrorMeasurer();

};


#endif
