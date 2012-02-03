#ifndef __MSEMEASURER_HPP__
#define __MSEMEASURER_HPP__
/*!
 * \file MSEMeasurer.hpp
 * Header of the MSEMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <stdexcept>



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
  ~MSEMeasurer();

};


#endif
