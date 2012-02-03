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
  std::list<Connection*> inputConnections;
  /*! Output connections*/
  std::list<Connection*> outputConnections;
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
   * Get input connections.
   * \return Input connections.
   */
  std::list<Connection*> getInputConnections() const;

  /*!
   * Get output connections.
   * \return Output connections.
   */
  std::list<Connection*> getOutputConnections() const;

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
   * Set input connections.
   * \param connections Input connections.
   */
  void setInputConnections(std::list<Connection*> connections);

  /*!
   * Set output connections.
   * \param connections  Output connections.
   */
  void setOutputConnections(std::list<Connection*> connections);

  /*!
   * Add input connections.
   * \param _connection Input connections.
   */
  void addInputConnections(Connection* _connection);

  /*!
   * Add output connections.
   * \param _connection Output connections.
   */
  void addOutputConnections(Connection* _connection);


  /*! 
   * Forward a feature vector.
   */
  virtual void forward()=0;

  /*!
   * Backward propagation of error.
   * \param deltas Errors from above layers.
   * \return Error vector of this layer.
   */
  virtual void backward(ErrorVector deltas)=0;

  /*!
   * Weight a signal coming in a neuron with respect to the weight matrix of this neuron.
   * \param signal Input signal.
   * \param weights Weight matrix.
   * \return Dot product.
   */
  realv signalWeighting(FeatureVector signal, cv::Mat weights);

  /*!
   * Destructor.
   */
  ~Layer();

};


#endif
