#ifndef __ERRORMEASURER_HPP__
#define __ERRORMEASURER_HPP__
/*!
 * \file ErrorMeasurer.hpp
 * Header of the ErrorMeasurer class.
 * \author Luc Mioulet
 */

#include "../../General.hpp"
#include "../../dataset/FeatureVector.hpp"
#include "../../dataset/ErrorVector.hpp"

/*!
 * \class ErrorMeasurer
 * Measure errors between a machine output vector and the data target.
 */
class ErrorMeasurer {
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  ErrorMeasurer();
  
  /*!
   * Measure error per unit.
   * \param ouput Machine output feature Vector.
   * \param target Dataset result to match.
   * \return The error vector. Has the same length as the output/target.
   */
  virtual ErrorVector errorPerUnit(FeatureVector _output, FeatureVector _target) = 0;

  /*!
   * Measure error per unit.
   * \param ouput Machine output feature Vector.
   * \param target Dataset result to match.
   * \return The total error.
   */
  virtual realv totalError(FeatureVector _output, FeatureVector _target) = 0;

  /*!
   * Destructor.
   */
  ~ErrorMeasurer();

};


#endif
