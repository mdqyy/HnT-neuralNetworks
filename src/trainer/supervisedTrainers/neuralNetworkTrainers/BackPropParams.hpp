#ifndef __BACKPROPPARAMS_HPP__
#define __BACKPROPPARAMS_HPP__
/*!
 * \file BackPropParams.hpp
 * Header of the BackPropParams class.
 * \author Luc Mioulet
 */

#include "../../../General.hpp"
#include <opencv/cv.h>

#define BP_CLASSIFICATION 0
#define BP_REGRESSION 1
#define BP_AUTOENCODER 2

/*!
 * \class BackPropParams
 * Description
 */
class BackPropParams {
 private :

 protected:
  /*! Do stochastic learning (randomize sequence selection).*/
  bool doStochastic;

  /*! Learning rate for weight change. */
  realv learningRate;

  /*! Learning rate decrease between each step. */
  realv learningRateDecrease;

  /*! Maximum number of iterations */
  uint maxIterations;

  /*! Minimum error change between two steps*/
  realv minError;
  
  /*! Minimum error change between two iterations */
  realv minChangeError;

  /*! Validation modulo steps */
  uint validationSteps;

  /*! Problem type (BP_CLASSIFICATION, BP_REGRESSION, BP_AUTOENCODER) */
  int task;
  

 public:

  /*!
   * Default constructor.
   */
  BackPropParams(bool _doStochastic=true,realv _learningRate=0.1, realv _learningRateDecrease=0.95, uint _maxIterations=10, realv _minError=0.01, realv _minChangeStop=1.0e-9, uint _validationSteps=0, int _task=0);

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
   * Get the validation step intervals.
   * \return Validation steps spacing.
   */
  uint getValidationSteps();
  
  /*!
   * Get the task.
   * \return The task (BP_CLASSIFICATION, BP_REGRESSION, BP_AUTOENCODER).
   */
  int getTask();

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
   * Set validation steps (a validation process will take place every time modulo(iteration,_validationSteps)=0).
   * \param _validationSteps Validation steps.
   */
  void setValidationSteps(uint _validationSteps);

  /*!
   * Set the task.
   * \param _task Task to be achieved (BP_CLASSIFICATION, BP_REGRESSION, BP_AUTOENCODER).
   */
  void setTask(int _Task);

  /*!
   * Destructor.
   */
  ~BackPropParams();

};


#endif
