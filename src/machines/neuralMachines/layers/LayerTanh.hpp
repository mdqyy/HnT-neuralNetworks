#ifndef __LAYERTANH_HPP__
#define __LAYERTANH_HPP__
/*!
 * \file LayerTanh.hpp
 * Header of the LayerTanh class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"
#include <stdexcept>

/*!
 * \class LayerTanh
 * Hyperbolic tangent layer.
 */
class LayerTanh : public Layer {
 private :

 protected:

 public:

  LayerTanh();

  /*!
   * Parameter constructor.
   * \param _numUnits Number of units in the layer.
   * \param _name Name of the layer.
   */
  LayerTanh(uint _numUnits,std::string _name="tanh layer");

  /*!
   * Copy cnstructor.
   * \param _clth Layer to copy.
   */
  LayerTanh(const LayerTanh& _clth);

  /*!
   * Clone a layer
   * \return Pointer to a clone.
   */
  virtual LayerTanh* clone() const;

  /*! 
   * Forward a feature vector.
   */
  virtual void forward();
  
  /*!
   * Backward propagation of error.
   * \param _output Calculate errors as an output layer.
   * \param _target Target of the output layer.
   */
  virtual void backwardDeltas(bool _output=false, FeatureVector _target=FeatureVector(0));

  /*!
   * Backward propagation of weight changes.
   * \param _learningRate Weight change rate.
   */
  virtual void backwardWeights(realv _learningRate);

  /*!
   * Destructor.
   */
  ~LayerTanh();


  /*!
   * Output stream fo
   * \param os Output stream.
   * \param l Layer.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const LayerTanh& l);
};


#endif
