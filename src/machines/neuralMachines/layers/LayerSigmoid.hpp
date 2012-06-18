#ifndef __LAYERSIGMOID_HPP__
#define __LAYERSIGMOID_HPP__
/*!
 * \file LayerSigmoid.hpp
 * Header of the LayerSigmoid class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"
#include <stdexcept>

/*!
 * \class LayerSigmoid
 * Hyperbolic tangent layer.
 */
class LayerSigmoid : public Layer {
 private :

 protected:

 public:

  LayerSigmoid();

  /*!
   * Parameter constructor.
   * \param _numUnits Number of units in the layer.
   * \param _name Name of the layer.
   * \param _recurrent Activate recurrency.
   */
  LayerSigmoid(uint _numUnits,std::string _name="sigmoid_layer", bool _recurrent=false);
  
  /*!
   * Copy constructor.
   * \param _cls Layer to copy.
   */
  LayerSigmoid(const LayerSigmoid& _cls);

  
  /*!
   * Clone a layer
   * \return Pointer to a clone.
   */
  virtual LayerSigmoid* clone() const;

  /*!
   * Get the layer type.
   * \return Layer type.
   */
  int getLayerType() const;

  /*! 
   * Forward a feature vector.
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
  // void backwardDeltas(bool _output=false, FeatureVector _target=FeatureVector(0));

  /*!
   * Backward propagation of weight changes.
   * \param _learningRate Weight change rate.
   */
  // void backwardWeights(realv _learningRate);

  /*!
   * Destructor.
   */
  virtual ~LayerSigmoid();

  /*!
   * Output stream fo
   * \param os Output stream.
   * \param l Layer.
   * \return Output stream.
   */
  //friend std::ostream& operator<<(std::ostream& os, const LayerSigmoid& l);
  void print(std::ostream& _os) const;

  /*!
   * File output stream.
   * \param ofs Output file stream.
   * \param l Sigmoid layer.
   * \return File Output stream.
   */
  friend std::ofstream& operator<<(std::ofstream& ofs, const LayerSigmoid& l);

  /*!
   * File input stream.
   * \param ifs Input file stream.
   * \param l Sigmoid layer.
   * \return File Input stream.
   */
    friend std::ifstream& operator>>(std::ifstream& _ifs, LayerSigmoid& _l);
};


#endif
