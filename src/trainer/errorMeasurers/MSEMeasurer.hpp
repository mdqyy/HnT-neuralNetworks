#ifndef __MSEMEASURER_HPP__
#define __MSEMEASURER_HPP__
/*!
 * \file MSEMeasurer.hpp
 * Header of the MSEMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <stdexcept>
#include <iostream>

/*!
 * \class MSEMeasurer
 * Measure errors between a machine output vector and the data target.
 */
class MSEMeasurer : public ErrorMeasurer {
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  MSEMeasurer();
  
  /*!
   * Get error.
   * \return Mean square error.
   */
  realv getError();

  /*!
   * Get error per unit
   * \return Mean square error per unit.
   */
  ErrorVector getErrorPerUnit();
  
  /*!
   * Measure error per unit.
   * \param ouput Machine output feature Vector.
   * \param target Dataset result to match.
   * \return The error vector. Has the same length as the output/target.
   */
  virtual ErrorVector errorPerUnit(FeatureVector _output, FeatureVector _target);

  /*!
   * Measure total mean square error of all units.
   * \param ouput Machine output feature Vector.
   * \param target Dataset result to match.
   * \return The total error.
   */
  virtual realv totalError(FeatureVector _output, FeatureVector _target);

  /*!
   * Destructor.
   */
  ~MSEMeasurer();

  /*!
   * Output stream operator.
   * \param os Output stream.
   * \param mse Error measurer.
   * \return Output stream.
   */
  friend std::ostream& operator<< (std::ostream& os, MSEMeasurer& mse);

};


#endif
