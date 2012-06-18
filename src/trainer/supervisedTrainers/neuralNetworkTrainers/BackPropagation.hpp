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


 protected:
  BackPropParams bpp;
  std::vector<realv> errorPerIteration;
  
  ErrorVector calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta);
  ErrorVector calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives);

  /*!
   * Update connection of the CTC input weight matrix.
   * \param _connection Connection input of the weight matrix.
   * \param _deltas Derivatives.
   * \param _learningRate Update connection rate.
   */
  void updateConnection(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate);
  realv measureSampleError(FeatureVector networkOutput, FeatureVector target);
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
   * Backward errors from one target.
   * \param _target Target feature vector.
   * \param _learningRate Learning rate for weight changes.
   */
  void backward(FeatureVector _target, realv _learningRate);

  /*!
   * Destructor.
   */
  ~BackPropagation();

};


#endif
