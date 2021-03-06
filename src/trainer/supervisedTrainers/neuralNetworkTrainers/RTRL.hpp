#ifndef __RTRL_HPP__
#define __RTRL_HPP__
/*!
 * \file RTRL.hpp
 * Header of the RTRL (Real-Time Recurrent Learning) class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include "../../errorMeasurers/MSEMeasurer.hpp"
#include "../../errorMeasurers/ClassificationErrorMeasurer.hpp"
#include "NeuralNetworkTrainer.hpp"
#include "RTRL.hpp"
#include <math.h>

/*!
 * \class RTRL
 * Description
 */
class RTRL : public NeuralNetworkTrainer {
 private :


 protected:
  RTRLParams bpp;
  std::vector<realv> errorPerIteration;
  
  /*!
   * p matrix linked to weights
   */
  vector< vector<Mat> > sensitivity;

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
   * \param _bpparams RTRL parameters.
   */
  RTRL(NeuralNetwork& _neuralNet, SupervisedDataset& _data, BackPropParams& _bpparams, Mask& _featureMask, Mask& _indexMask);

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

  /*
   * Reset pWeights during learning.
   */
  void resetSensitivity();

  /*!
   * Destructor.
   */
  ~RTRL();

};


#endif
