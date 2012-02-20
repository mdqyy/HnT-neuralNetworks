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
   * \param _featureMask Feature mask.
   * \param _indexMask Sample index mask.
   * \param _bpparams Backpropagation parameters.
   */
  BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _data, BackPropParams& _bpparams, Mask& _featureMask, Mask& _indexMask);

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
