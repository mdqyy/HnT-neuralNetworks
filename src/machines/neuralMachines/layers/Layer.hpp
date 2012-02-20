#ifndef __LAYER_HPP__
#define __LAYER_HPP__
/*!
 * \file Layer.hpp
 * Header of the Layer class.
 * \author Luc Mioulet
 */

#include <list>
#include <assert.h>
#include "../../../General.hpp"
#include "../../../dataset/FeatureVector.hpp"
#include "../../../dataset/ErrorVector.hpp"
#include "../connections/Connection.hpp"
#include "../../Machine.hpp"

class Connection;

/*!
 * \class Layer
 * Abstract class.
 * Neural network layer.
 */
class Layer : public Machine {
 private :

 protected:
  /*! Number of units of this layer */
  uint numUnits;
  /*! Input connections*/
  Connection* inputConnection;
  /*! Output connections*/
  Connection* outputConnection;
  /*!Output signal of the layer */
  FeatureVector outputSignal;
  /*! Output error of the layer */
  ErrorVector deltas;
  
 public:

  /*!
   * Parameter constructor.
   * \param _numUnits Number of units.
   * \param _name Name of the layer.
   */
  Layer(uint _numUnits, std::string _name);
  
  /*!
   * Get the number of units.
   * \return Number of units in the network.
   */
  uint getNumUnits() const;
  
  /*!
   * Get input connection.
   * \return Input connections.
   */
  Connection* getInputConnection() const;

  /*!
   * Get output connection.
   * \return Output connection.
   */
  Connection* getOutputConnection() const;

  /*!
   * Get last output feature vector.
   * \return Feature vector.
   */
  FeatureVector getOutputSignal() const;

  /*!
   * Get last output error vector;
   * \return Error vector.
   */
  ErrorVector getErrorVector() const;

  /*!
   * Set error vector of the layer.
   * \param _deltas New error vector.
   * \remark Should only be used on the output layer.
   */
  void setErrorVector(ErrorVector _deltas);

  /*!
   * Set input connection.
   * \param connection Input connection.
   */
  void setInputConnection(Connection* _connection);

  /*!
   * Set output connection.
   * \param _connection Output connection.
   */
  void setOutputConnection(Connection* _connection);

  /*!
   * Forward a sequence.
   * \param _sequence Sequence of feature vectors.
   */
  virtual void forwardSequence(std::vector<FeatureVector> _sequence);

  /*! 
   * Forward a feature vector.
   */
  virtual void forward()=0;

  /*!
   * Backward propagation of error.
   * \param _output Calculate errors as an output layer.
   * \param _target Target of the output layer.
   */
  virtual void backwardDeltas(bool _output=false, FeatureVector _target=FeatureVector(0))=0;

  /*!
   * Backward propagation of weight changes.
   */
  virtual void backwardWeights(realv _learningRate)=0;

  /*!
   * Weight a signal coming in a neuron with respect to the weight matrix of this neuron.
   * \param _signal Input signal.
   * \param _weights Weight matrix.
   * \return Dot product.
   */
  realv signalWeighting(FeatureVector _signal, cv::Mat _weights);

  /*!
   * Weight an error signal coming in a neuron with respect to the weight matrix of this neuron.
   * \param _deltas Error signal.
   * \param _weights Weight matrix.
   * \return Dot product.
   */
  realv errorWeighting(ErrorVector _deltas, cv::Mat _weights);


  /*!
   * Destructor.
   */
  ~Layer();

};


#endif
