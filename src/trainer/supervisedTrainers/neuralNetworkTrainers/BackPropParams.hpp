#ifndef __BACKPROPPARAMS_HPP__
#define __BACKPROPPARAMS_HPP__
/*!
 * \file BackPropParams.hpp
 * Header of the BackPropParams class.
 * \author Luc Mioulet
 */

#include "../../../General.hpp"

/*!
 * \class BackPropParams
 * Description
 */
class BackPropParams {
 private :

 protected:
  bool doStochastic;
  realv learningRate;
  realv learningRateDecrease;

 public:

  /*!
   * Default constructor.
   */
  BackPropParams(bool _doStochastic=true,realv _learningRate=0.001, realv _learningRateDecrease=0.0);

  /*!
   * Do stochastic.
   * \return Do stochastic.
   */
  bool getDoStochastic();

  /*!
   * Get learning rate.
   * \return Weight change rate per iteration.
   */
  realv getLearningRate();

  /*!
   * Get Learning rate decrease.
   * \return Weight decrease.
   */
  realv getLearningRateDecrease();

  /*!
   * Set stochastic.
   * \param Do stochastic.
   */
  void setDoStochastic(bool _doStochastic);

  /*!
   * Set learning rate.
   * \param _learningRate New learning rate.
   */
  void setLearningRate(realv _learningRate);

  /*!
   * Set learning rate decrease.
   * \param _learningRateDecrease Learing rate decrease.
   */
  void setLearningRateDecrease(realv _learningRateDecrease);

  /*!
   * Destructor.
   */
  ~BackPropParams();

};


#endif
