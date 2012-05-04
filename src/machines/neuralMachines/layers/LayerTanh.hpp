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
  LayerTanh(uint _numUnits,std::string _name="tanh_layer");

  /*!
   * Copy cnstructor.
   * \param _clth Layer to copy.
   */
  LayerTanh(const LayerTanh& _clth);

  /*!
   * Clone a layer
   * \return Pointer to a clone.
   */
  LayerTanh* clone() const;

  /*!
   * Get the layer type.
   * \return Layer type.
   */
  virtual int getLayerType() const;

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
   * Backward propagation of error.
   * \param _output Calculate errors as an output layer.
   * \param _target Target of the output layer.
   */
  void backwardDeltas(bool _output=false, FeatureVector _target=FeatureVector(0));

  /*!
   * Backward propagation of weight changes.
   * \param _learningRate Weight change rate.
   */
  void backwardWeights(realv _learningRate);

  /*!
   * Destructor.
   */
  virtual ~LayerTanh();

  /*!
   * Output stream fo
   * \param os Output stream.
   * \param l Layer.
   * \return Output stream.
   */
  //friend std::ostream& operator<<(std::ostream& os, const LayerTanh& l);
  void print(std::ostream& _os) const;

  /*!
   * File output stream.
   * \param ofs Output file stream.
   * \param l Tanh layer.
   * \return File Output stream.
   */
  friend std::ofstream& operator<<(std::ofstream& ofs, const LayerTanh& l);

  /*!
   * File input stream.
   * \param ifs Input file stream.
   * \param l Tanh layer.
   * \return File Input stream.
   */
  friend std::ifstream& operator>>(std::ifstream& ifs, LayerTanh& l);
};


#endif
