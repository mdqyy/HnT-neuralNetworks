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
  ValueVector meanVector;
  /*! Standard deviation */
  ValueVector stdevVector;


 public:

  InputLayer();

  /*!
   * Parameter constructor.
   * \param _numUnits Number of units in the layer
   * \param _mean Mean of normalization.
   * \param _stdev Standard deviation.
   * \param _name Name of the layer.
   */
  InputLayer(uint _numUnits, ValueVector _mean, ValueVector _stdev, std::string _name="input_layer");

  /*! 
   * Copy an existing InputLayer.
   * \param _cil Input Layer to copy.
   */
  InputLayer(const InputLayer& _cil);

  /*!
   * Clone a layer
   * \return Pointer to a clone.
   */
  InputLayer* clone() const;

  /*!
   * Get the layer type.
   * \return Layer type.
   */
  int getLayerType() const;

  /*!
   * Get mean value.
   * \return The mean value.
   */
  ValueVector getMean() const;

  /*!
   * Get the standard deviation.
   * \return The standard deviation.
   */
  ValueVector getStandardDeviation() const;

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
  void forward();

  /*! 
   * Forward a feature vector.
   * \param _signal Input signal.
   * \return Output feature vector. 
   */
  void forward(FeatureVector _signal);

  /*! 
   * Process the derivative for the layer.
   * \return A value vector containing the derivative.
   */
  ValueVector getDerivatives() const;

  /*!
   * Backward propagation of error.
   * \param _output Calculate errors as an output layer.
   * \param _target Target of the output layer.
   */
  //void backwardDeltas(bool _output=true, FeatureVector _target=FeatureVector(0));


  /*!
   * Backward propagation of weight changes.
   * \params _learningRate Weight change rate.
   */
  //void backwardWeights(realv _learningRate);

  /*!
   * Destructor.
   */
  virtual ~InputLayer();

  /*!
   * Output stream.
   * \param os Output stream.
   * \param l Layer.
   * \return Output stream.
   */
  // friend std::ostream& operator<<(std::ostream& os, const InputLayer& l);
  void print(std::ostream& _os) const;

  /*!
   * File output stream.
   * \param ofs Output file stream.
   * \param l Input layer.
   * \return File Output stream.
   */
  friend std::ofstream& operator<<(std::ofstream& ofs, const InputLayer& l);

  /*!
   * File input stream.
   * \param ifs Input file stream.
   * \param l Input layer.
   * \return File Input stream.
   */
  friend std::ifstream& operator>>(std::ifstream& ifs, InputLayer& l);
};


#endif
