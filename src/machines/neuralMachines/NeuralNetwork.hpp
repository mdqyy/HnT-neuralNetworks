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
#include "connections/Connection.hpp"
#include "../../General.hpp"
#include "../../dataset/FeatureVector.hpp"
#include "../../dataset/ErrorVector.hpp"
#include <list>


/*!
 * \class NeuralNetwork
 * Description
 */
class NeuralNetwork : public NeuralMachine {
 private :

 protected:
  /*! Input layer */
  InputLayer& input; 
  /*! Hidden layer */
  std::list<Layer*> hiddenLayers; 
  /*! Output layer */
  Layer& output; 
  /*! Connections between layers */
  std::list<Connection*> connections;
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
  NeuralNetwork(InputLayer& _input, std::list<Layer*> _hidden, Layer& _output, std::list<Connection*> _connections, bool _forward=true,std::string _name="neural network");

  /*!
   * Get input layer.
   * \return Input layer.
   */
  InputLayer& getInputLayer();

  /*!
   * Get hidden layers.
   * \return Hidden layers.
   */ 
  std::list<Layer*> getHiddenLayers();

  /*!
   * Get output layer.
   * \return Output layer.
   */
  Layer& getOutputLayer();

  /*!
   * Get input signal.
   * \return Input signal.
   */
  FeatureVector getInputSignal();

  /*!
   * Get output signal.
   * \return Output signal.
   */
  FeatureVector getOutputSignal();

  /*!
   * Get the forward 
   * \return True if the network reads the sequence in a forward manner. False if it reads it backwards.
   */
  bool isForward();

  /*!
   * Set an input layer.
   * \param _input Input layer.
   */
  void setInputLayer(InputLayer& _input);

  /*!
   * Set hidden layers.
   * \param _hidden Hidden layers.
   */
  void setHiddenLayers(std::list<Layer*> _hidden);

  /*!
   * Set an output layer.
   * \param _output Output layer.
   */
  void setOutputLayer(Layer& _output);

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
