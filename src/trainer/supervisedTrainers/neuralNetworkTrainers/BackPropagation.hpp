#ifndef __BACKPROPAGATION_HPP__
#define __BACKPROPAGATION_HPP__
/*!
 * \file BackPropagation.hpp
 * Header of the BackPropagation class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include "../../errorMeasurers/MSEMeasurer.hpp"
#include "../../errorMeasurers/ClassificationErrorMeasurer.hpp"
#include "NeuralNetworkTrainer.hpp"
#include "BackPropParams.hpp"
#include <math.h>

/*!
 * \class BackPropagation
 * Description
 */
class BackPropagation : public NeuralNetworkTrainer {
 private :
  BackPropParams bpp;
  std::vector<realv> errorPerIteration;

 protected:

 public:

  /*!
   * Parameter constructor.
   * \param _neuralNet NeuralNetwork to train.
   * \param _data Supervised dataset to use for learning.
   * \param _doStochastic Activate stochastic (random) forwarding of data.
   */
  BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _data, CrossValidationParams& _cvParams, BackPropParams& _bpparams);

  /*!
   * Parameter constructor to set a test and a validation dataset.
   * \param _machine Neural network to train.
   * \param _trainData Supervised dataset used for learning.
   * \param _validationData Supervised dataset used for validation.
   * \param _trainData Supervised dataset used for testing.
   * \param _cvParams Cross Validation information.
   * \param _bpparams Activate stochastic (random) forwarding of data.
   */
  BackPropagation(NeuralNetwork& _machine, SupervisedDataset& _trainData, SupervisedDataset& _validationData, SupervisedDataset& _testData, CrossValidationParams& _cvParams, BackPropParams& _bpparams);

  /*!
   * Train the neural network.
   */
  virtual void train();

  /*!
   * Train the neural network on one iteration.
   */
  void trainOneIteration();

  /*!
   * Destructor.
   */
  ~BackPropagation();

};


#endif
