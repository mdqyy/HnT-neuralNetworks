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
#include "../../../Clonable.hpp"
#include "../../../dataset/FeatureVector.hpp"
#include "../../../dataset/ErrorVector.hpp"
#include "../connections/Connection.hpp"
#include "../../Machine.hpp"
#include <boost/shared_ptr.hpp>

#define LAYER 0
#define LAYER_INPUT 1
#define LAYER_SIGMOID 2
#define LAYER_TANH 3
#define LAYER_SOFTMAX 4
#define LAYER_CTC 5

class Connection;


/*!
 * \class Layer
 * Abstract class.
 * Neural network layer.
 */
class Layer : public Machine, public Clonable {
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
  /*! Input signal from network (not the actual recurrent input signal) */
  FeatureVector networkInputSignal; 
  /*! Complete input signal (i.e taking into account recurrency) */
  FeatureVector inputSignal;
  /*! Boolean to indicate use of recurrent connections */
  bool recurrent;
  /*! Vector of input vectors */
  std::vector<FeatureVector> inputSignals;
  /*! Vector of output vectors */
  std::vector<FeatureVector> outputSignals;

 public:

  Layer();

  /*!
   * Parameter constructor.
   * \param _numUnits Number of units.
   * \param _name Name of the layer.
   */
  Layer(uint _numUnits, std::string _name, bool _recurrent=false);

  /*!
   * Copy constructor.
   * \param _cl Layer to copy.
   */
  Layer(const Layer& _cl);
  
  /*!
   * Clone method.
   * \return A clone of a layer.
   */
  virtual Layer* clone() const = 0;
  
  /*!
   * Get the layer type.
   * \return Layer type.
   */
  virtual int getLayerType() const = 0;
  
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
   * Get output signals.
   * \return Vector of feature vectors.
   */
  std::vector<FeatureVector> getOutputSignals() const;

  /*!
   * Get input signals
   * \return Vector of feature vectors.
   */
  std::vector<FeatureVector> getInputSignals() const;

  /*!
   * Get recurrency information.
   * \return True if recurrent.
   */
  bool isRecurrent() const;

  /*!
   * Get last output feature vector.
   * \return Feature vector.
   */
  FeatureVector getOutputSignal() const;

  /*!
   * Get the last input signal from the network.
   * \return The input signal from the network.
   */
  FeatureVector getNetworkInputSignal() const;

  /*!
   * Create the input signal for the layer the network.
   * If it is a recurrent layer it will concatenate actual input and previous output.
   * If it is a regular layer it will be the same as the network input signal.
   * \return The input signal from the network.
   */
  FeatureVector createInputSignal() const;

  /*!
   * Get Input signal.
   * \return Layer input signal.
   */
  FeatureVector getInputSignal() const;

  /*!
   * Get last output error vector;
   * \return Error vector.
   */
  ErrorVector getErrorVector() const;

  /*!
   * Set number of units.
   * \remark Do not use after having connected this.
   * \param _numUnits Number of units.
   */
  void setNumUnits(uint _numUnits);

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

  /*
   * Set recurrency.
   * \param _state Turn on or off recurrency.
   */
  void setRecurent(bool _state);

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
   * Forward a feature vector.
   * \param _signal Input signal.
   * \return Output feature vector. 
   */
  virtual void forward(FeatureVector _signal)=0;

  /*! 
   * Process the derivative for the layer.
   * \return A value vector containing the derivative.
   */
  virtual ValueVector getDerivatives() const =0;

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
   * Clear input and output signals from data.
   */
  void reset();

  /*!
   * Print data concerning the object.
   * \param _os Output file stream.
   */
  virtual void print(std::ostream& _os) const=0;

  /*!
   * Destructor.
   */
  virtual ~Layer();
};

typedef boost::shared_ptr<Layer> LayerPtr;

#endif
