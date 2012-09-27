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
#include <math.h>


/*!
 * \class ErrorMeasurer
 * Measure errors between a machine output vector and the data target.
 */
class ErrorMeasurer {
 private :

 protected:
  /*! Total error */
  realv err;

  /*! Error per unit */
  ErrorVector errPerUnit;

 public:

  /*!
   * Default constructor.
   */
  ErrorMeasurer();

  /*!
   * Get total error.
   * \return Previously calculate total error.
   */
  realv getError();

  /*!
   * Get error per unit.
   * \return Error per unit vector.
   */
  ErrorVector getErrorPerUnit();

  /*!
   * Set a new value for error.
   * \param _error New error value.
   */
  void setError(realv _error);

  /*!
   * Set new values for the error per units.
   * \param _errPerUnit New error per units.
   */
  void setErrorPerUnit(ErrorVector _errPerUnit);
  

  /*!
   * Process all errors.
   * \param _ouput Machine output feature Vector.
   * \param _target Dataset result to match.
   */
  virtual void processErrors(FeatureVector _output, FeatureVector _target)=0;

  /*!
   * Measure error per unit.
   * \param _ouput Machine output feature Vector.
   * \param _target Dataset result to match.
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
