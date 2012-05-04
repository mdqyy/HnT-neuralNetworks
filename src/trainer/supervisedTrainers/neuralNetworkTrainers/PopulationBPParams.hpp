#ifndef __POPULATIONBPPARAMS_HPP__
#define __POPULATIONBPPARAMS_HPP__
/*!
 * \file PopulationBPParams.hpp
 * Header of the PopulationBPParams class.
 * \author Luc Mioulet
 */

#include "../../../General.hpp"
#include <opencv/cv.h>

#define BP_CLASSIFICATION 0
#define BP_REGRESSION 1
#define BP_AUTOENCODER 2

/*!
 * \class PopulationBPParams
 * Description
 */
class PopulationBPParams {
 private :

 protected:
  /*! Learning rate for weight change. */
  realv learningRate;

  /*! Learning rate decrease between each step. */
  realv learningRateDecrease;

  /*! Maximum number of iterations */
  uint maxIterations;

  /*! Maximum number of networks to train per iteration*/
  uint maxTrained;
  
  /*! Error ratio between best classifier and worst selected for training.*/
  realv errorToFirst;

  /*! Error ratio increase */
  realv errorToFirstIncrease;

 public:

  /*!
   * Default constructor.
   */
  PopulationBPParams(realv _learningRate=0.001, realv _learningRateDecrease=0.95, uint _maxIterations=10, uint _maxTrained=5, realv _errorToFirst=0.5,realv errorToFirstIncrease=1.1);

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
   * Get the maxmimum number of networks to train at each iteration.
   * \return Maximum of netwoks trained.
   */
  uint getMaxTrained();

  /*!
   * Get error to best network selected.
   * \return Error to best network.
   */
  realv getErrorToFirst();

  /*!
   * Get ratio increase to best network selected.
   * \return Increase ratio to best network.
   */
  realv getErrorToFirstIncrease();
  
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
   * Set maximum of networks to train.
   * \param _maxTrained Maximum of networks to train.
   */
  void setMaxTrained(uint _maxTrained);

  /*!
   * Set the error to first ratio.
   * \param _errorToFirst Error to first.
   */
  void setErrorToFirst(realv _errorToFirst);

  /*!
   * Set the increase for the ratio to first.
   * \param _errorToFirstIncrease Increase.
   */
  void setErrorToFirstIncrease(realv _errorToFirstIncrease);

  /*!
   * Destructor.
   */
  ~PopulationBPParams();

};


#endif
