#ifndef __SUPERVISEDTRAINER_HPP__
#define __SUPERVISEDTRAINER_HPP__
/*!
 * \file SupervisedTrainer.hpp
 * Header of the SupervisedTrainer class.
 * \author Luc Mioulet
 */

#include "../Trainer.hpp"
#include "../../dataset/supervised/SupervisedDataset.hpp"
#include "CrossValidationParams.hpp"


/*!
 * \class SupervisedTrainer
 * Abstract class for training supervised machines.
 * Includes cross validation.
 */
class SupervisedTrainer : public Trainer{
 private :

 protected:
  SupervisedDataset& trainData;
  SupervisedDataset& validationData;
  SupervisedDataset& testData;
  CrossValidationParams& cvParams;

 public:

  /*!
   * Parameter constructor.
   * \param _machine Machine to train.
   * \param _data Supervised dataset used for learning.
   * \param _cvParams Cross Validation information.
   */
  SupervisedTrainer(Machine& _machine, SupervisedDataset& _data, CrossValidationParams& _cvParams);

  /*!
   * Parameter constructor to set a test and a validation dataset.
   * \param _machine Machine to train.
   * \param _trainData Supervised dataset used for learning.
   * \param _validationData Supervised dataset used for validation.
   * \param _trainData Supervised dataset used for testing.
   * \param _cvParams Cross Validation information.
   */
  SupervisedTrainer(Machine& _machine, SupervisedDataset& _trainData, SupervisedDataset& _validationData, SupervisedDataset& _testData, CrossValidationParams& _cvParams);
  
  /*! 
   * Get the validation dataset.
   * \return The validation dataset.
   */
  SupervisedDataset& getValidationDataset() const;

  /*! 
   * Get the test dataset.
   * \return The test dataset.
   */
  SupervisedDataset& getTestDataset() const;

  /*! 
   * Get cross validation parameters.
   * \return The cross validation information.
   */
  CrossValidationParams& getCrossValidationParams() const;

  /*!
   * Train the machine.
   */
  virtual void train() =0;    

  /*!
   * Destructor.
   */
  ~SupervisedTrainer();

};


#endif
