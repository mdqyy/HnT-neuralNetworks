#ifndef __BACKPROPPARAMS_HPP__
#define __BACKPROPPARAMS_HPP__
/*!
 * \file BackPropParams.hpp
 * Header of the BackPropParams class.
 * \author Luc Mioulet
 */

#include "../../../General.hpp"
#include <opencv/cv.h>

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
  uint maxIterations;
  realv minError;
  realv minChangeError;

 public:

  /*!
   * Default constructor.
   */
  BackPropParams(bool _doStochastic=true,realv _learningRate=0.001, realv _learningRateDecrease=0.0, uint _maxIterations=10, realv _minError=0.1, realv _minChangeStop=0.001);

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
   * Get maximum number of iterations.
   * \return Maximum number of iterations.
   */
  uint getMaxIterations();

  /*!
   * Get the minimum error to be achieved.
   * \return Minimum error to be achieved.
   */
  realv getMinError();

  /*!
   * Get minimum error change for which the backpropagation shoud be stopped.
   * \return Minimum error change.
   */
  realv getMinChangeError();

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
   * Set maximum iterations.
   * \param _maxIterations Maximum number of iterations.
   */
  void setMaxIterations(uint _maxIterations);

  /*!
   * Set minimum error to be achieved.
   * \param _minError Minimum error.
   */
  void setMinError(realv _minError);

  /*!
   * Set the minimum change in error before backpropagation is stopped.
   * \param _minChangeError Minimum error change between two iterations.
   */
  void setMinChangeError(realv _minChangeError);

  /*!
   * Destructor.
   */
  ~BackPropParams();

};


#endif
