#ifndef __INPUTLAYER_HPP__
#define __INPUTLAYER_HPP__
/*!
 * \file InputLayer.hpp
 * Header of the InputLayer class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"
#include "../../../dataset/ValueVector.hpp"

/*!
 * \class InputLayer
 * The input layer should be the first layer of every neural network.
 * It reduces and centers the data given (if these parameters are set).
 */
class InputLayer : public Layer{
 private :

 protected:
  /*! Mean */
  ValueVector mean;
  /*! Standard deviation */
  ValueVector stdev;
  /*! Input signal */
  FeatureVector inputSignal;

 public:

  /*!
   * Parameter constructor.
   * \param _numUnits Number of units in the layer
   * \param _mean Mean of normalization.
   * \param _stdev Standard deviation.
   * \param _name Name of the layer.
   */
  InputLayer(uint _numUnits, ValueVector _mean, ValueVector _stdev, std::string _name="input layer");

  /*!
   * Get mean value.
   * \return The mean value.
   */
  ValueVector getMean();

  /*!
   * Get the standard deviation.
   * \return The standard deviation.
   */
  ValueVector getStandardDeviation();

  /*!
   * Get the last input signal.
   * \return The input signal.
   */
  FeatureVector getInputSignal();

  /*!
   * Set the mean.
   * \param _mean New mean value.
   */
  void setMean(ValueVector _mean);

  /*!
   * Set the standard deviation.
   * \param _stdev New mean value.
   */
  void setStandardDeviation(ValueVector _stdev);

 /*! 
   * Forward a feature vector from a previous layer.
   * \return Output feature vector. 
   */
   virtual void forward();

  /*! 
   * Forward a feature vector.
   * \param _signal Input signal.
   * \return Output feature vector. 
   */
  void forward(FeatureVector _signal);

  /*!
   * Backward propagation of error.
   * \param _output Calculate errors as an output layer.
   * \param _target Target of the output layer.
   */
  virtual void backwardDeltas(bool _output=true, FeatureVector _target=FeatureVector(0));


  /*!
   * Backward propagation of weight changes.
   * \params _learningRate Weight change rate.
   */
  virtual void backwardWeights(realv _learningRate);

  /*!
   * Destructor.
   */
  ~InputLayer();

  /*!
   * Output stream fo
   * \param os Output stream.
   * \param l Layer.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const InputLayer& l);

};


#endif
