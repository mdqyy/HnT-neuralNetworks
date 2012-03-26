#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__
/*!
 * \file NeuralNetwork.hpp
 * Header of the NeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralMachine.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/LayerSigmoid.hpp"
#include "connections/Connection.hpp"
#include "../../General.hpp"
#include "../../Clonable.hpp"
#include "../../dataset/FeatureVector.hpp"
#include "../../dataset/ErrorVector.hpp"
#include <vector>


/*!
 * \class NeuralNetwork
 * Description
 */
class NeuralNetwork : public NeuralMachine, public Clonable {
 private :

 protected:
  /*! Input layer */
  InputLayer* input; 
  /*! Hidden layer */
  std::vector<Layer*> hiddenLayers; 
  /*! Output layer */
  Layer* output; 
  /*! Connections between layers */
  std::vector<Connection*> connections;
  /*! Input signal */
  FeatureVector inputSignal;
  /*! Output signal */
  FeatureVector outputSignal;
  /*! Forward or backward sequence processing. */
  bool readForward;
    
 public:

  /*! 
   * Parameter constructor. 
   * \param _input Input layer.
   * \param _hidden Hidden layers.
   * \param _output Output layer.
   * \param _connections Connections between layers.
   * \param _forward Controlling the sequence processing of the machine.
   * \param _name Machine name.
   */
  NeuralNetwork(InputLayer* _input, std::vector<Layer*> _hidden, Layer* _output, std::vector<Connection*> _connections, bool _forward=true,std::string _name="neural network");

  /*!
   * Copy constructor.
   * \param _cnn Neural network to copy.
   */
  NeuralNetwork(const NeuralNetwork& _cnn);

  /*!
   * Clone an instance of a neural network.
   * \return A pointer to the clone.
   */
  NeuralNetwork* clone() const;
  

  /*!
   * Get input layer.
   * \return Input layer.
   */
  InputLayer* getInputLayer() const;

  /*!
   * Get hidden layers.
   * \return Hidden layers.
   */ 
  std::vector<Layer*> getHiddenLayers() const;

  /*!
   * Get output layer.
   * \return Output layer.
   */
  Layer* getOutputLayer() const;

  /*!
   * Get input signal.
   * \return Input signal.
   */
  FeatureVector getInputSignal() const;

  /*!
   * Get output signal.
   * \return Output signal.
   */
  FeatureVector getOutputSignal() const;

  /*!
   * Get Connections in the neural network.
   * \return Connections.
   */
  std::vector<Connection*> getConnections() const;

  /*!
   * Get the forward 
   * \return True if the network reads the sequence in a forward manner. False if it reads it backwards.
   */
  bool isForward() const;

  /*!
   * Set an input layer.
   * \param _input Input layer.
   */
  void setInputLayer(InputLayer* _input);

  /*!
   * Set hidden layers.
   * \param _hidden Hidden layers.
   */
  void setHiddenLayers(std::vector<Layer*> _hidden);

  /*!
   * Set an output layer.
   * \param _output Output layer.
   */
  void setOutputLayer(Layer* _output);

  /*!
   * Forward a sequence.
   * \param _sequence Sequence of feature vectors.
   */
  virtual void forwardSequence(std::vector<FeatureVector> _sequence);

  /*!
   * Forward a feature vector.
   * \param signal Input feature vector.
   */
  void forward(FeatureVector _signal);

  /*!
   * Backward an error vector.
   * \param _target Neural network target.
   * \param _learningRate Weight change rate.
   */
  void backward(FeatureVector _target, realv _learningRate);
  

  /*!
   * Destructor.
   */
  ~NeuralNetwork();

};


#endif
