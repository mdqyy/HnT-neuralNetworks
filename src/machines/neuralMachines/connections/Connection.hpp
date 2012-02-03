#ifndef __CONNECTION_HPP__
#define __CONNECTION_HPP__
/*!
 * \file Connection.hpp
 * Header of the Connection class.
 * \author Luc Mioulet
 */

#include <stdlib.h>
#include <iostream>
#include "../layers/Layer.hpp"
#include "../../../General.hpp"
#include <opencv/cv.h>

class Layer;

/*!
 * \class Connection
 * Class used to connect two layers together. This is a full connection type class.
 */
class Connection {
 private :

 protected:
  /*! Pointer to input Layer */
  Layer& from; 
  /*! Pointer to output Layer */
  Layer& to; 
 /*! Weight matrix is organised as rows = outputs, cols = inputs, and is accessed as weights[rows/outputs][lines/inputs] */
  cv::Mat weights;

 public:

  /*!
   * Parameter constructor.
   * Weight matrix will be initialized at 0 values.
   * \param _from Reference to the source layer.
   * \param _to Reference to the destination layer.
   * \param _seed Weight initialization.
   */
  Connection(Layer& _from, Layer& _to, uint _seed=42);

  /*!
   * Parameter constructor.
   * \param _from Pointer to the source layer.
   * \param _to Pointer to the destination layer.
   * \param _weight Weight matrix.
   */
  Connection(Layer& _from, Layer& _to, cv::Mat _weight);

  /*! 
   * Get weights.
   * \return Weight matrix.
   */
  cv::Mat getWeights() const;

  /*! 
   * Get input layer.
   * \return Input layer.
   */
  Layer& getInputLayer() const;

  /*! 
   * Get output layer.
   * \return Output layer.
   */
  Layer& getOutputLayer() const;

  /*!
   * Set weights.
   * \param _weights Weight matrix
   */
  void setWeights(cv::Mat _weights);

  /*!
   * Set input layer.
   * \param _input Input layer.
   */
  void setInputLayer(Layer& _input);

  /*!
   * Set output layer.
   * \param _output Output layer.
   */
  void setOutputLayer(Layer& _output);
  
  /*! 
   * Initialize weights according to a normal distribution.
   * \param seed Random seed.
   * \param mean Normal distribution mean
   * \param stdev Normal distribution standard deviation.
   */
  void initializeWeights(uint seed,realv mean=0, realv stdev=0.1);

  /*!
   * Get the weights concerning the input neuron i
   * \param i Input neuron.
   * \return Weight matrix.
   */
  cv::Mat getWeightsNeuron(int i);

  /*!
   * Forward call to next layer.
   * \todo parallel networks…
   */
  void forward();

  /*!
   * Destructor.
   */
  ~Connection();

  /*!
   * Output stream fo
   * \param os Output stream.
   * \param c Connection.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const Connection& c);
};


#endif
