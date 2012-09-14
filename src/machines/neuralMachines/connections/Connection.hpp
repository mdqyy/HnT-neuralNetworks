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
#include "../../../Clonable.hpp"
#include <opencv/cv.h>
#include <boost/shared_ptr.hpp>

class Layer;

/*!
 * \class Connection
 * Class used to connect two layers together. This is a full connection type class.
 */
class Connection : public Clonable{
 private :

 protected:
  /*! Pointer to input Layer */
  Layer* from; 
  /*! Pointer to output Layer */
  Layer* to; 
 /*! Weight matrix is organised as rows = outputs, cols = inputs, and is accessed as weights[rows/outputs][lines/inputs] */
  cv::Mat weights;

 public:

  /*!
   * Default constructor.
   */
  Connection();

  /*!
   * Parameter constructor.
   * Weight matrix will be initialized at 0 values.
   * \param _from Reference to the source layer.
   * \param _to Reference to the destination layer.
   * \param _seed Weight initialization.
   */
  Connection(Layer* _from, Layer* _to, uint _seed=42);

  /*!
   * Parameter constructor.
   * \param _from Pointer to the source layer.
   * \param _to Pointer to the destination layer.
   * \param _weight Weight matrix.
   */
  Connection(Layer* _from, Layer* _to, cv::Mat _weight);

  /*!
   * Copy constructor.
   * \remark You should change the connections via the set methods to avoid problems.
   * \param _c Connection to copy.
   */
  Connection(const Connection& _c);

  /*!
   * Clone an existing connection.
   * \remark You should change the layers via the set methods to avoid problems.
   * \return A pointer to the clone.
   */
  virtual Connection* clone() const;

  /*! 
   * Get weights.
   * \return Weight matrix.
   */
  cv::Mat getWeights() const;

  /*! 
   * Get input layer.
   * \return Input layer.
   */
  Layer* getInputLayer() const;

  /*! 
   * Get output layer.
   * \return Output layer.
   */
  Layer* getOutputLayer() const;

  /*!
   * Set weights.
   * \param _weights Weight matrix
   */
  void setWeights(cv::Mat _weights);

  /*!
   * Set input layer.
   * \param _input Input layer.
   */
  void setInputLayer(Layer* _input);

  /*!
   * Set output layer.
   * \param _output Output layer.
   */
  void setOutputLayer(Layer* _output);
  
  /*! 
   * Initialize weights according to a normal distribution.
   * \param _seed Random seed.
   * \param _mean Normal distribution mean
   * \param _stdev Normal distribution standard deviation.
   */
  void initializeWeights(uint _seed,realv _mean=0, realv _stdev=5.0);

  /*!
   * Get the weights concerning the input neuron i
   * \param _i Input neuron.
   * \return Weight matrix.
   */
  cv::Mat getWeightsToNeuron(int _i);

 /*!
   * Get the weights concerning the output neuron i
   * \param _i Input neuron.
   * \return Weight matrix.
   */
  cv::Mat getWeightsFromNeuron(int _i);

  /*!
   * Forward call to next layer.
   * \todo parallel networks…
   */
  void forward();

  /*!
   * Backward errors call to previous layer.
   * \param _output The previous layer is an output.
   */
  // void backwardDeltas(bool _output);

  /*!
   * Backward errors call to next layer.
   */
  //void backwardWeights(realv _learningRate);

  /*!
   * Destructor.
   */
  ~Connection();

  /*!
   * Output stream.
   * \param os Output stream.
   * \param c Connection.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const Connection& c);


  /*!
   * Output file stream.
   * \param ofs Output file stream.
   * \param c Connection.
   * \return Output file stream.
   */
  friend std::ofstream& operator<<(std::ofstream& ofs, const Connection& c);

  /*!
   * Input file stream.
   * \param ifs Input file stream.
   * \param c Connection.
   * \return Input file stream.
   */
  friend std::ifstream& operator>>(std::ifstream& ifs, Connection& c);
};

/*! ConnectionPtr definition */
typedef boost::shared_ptr<Connection> ConnectionPtr;

#endif
