#ifndef __PBDNN_HPP__
#define __PBDNN_HPP__
/*!
 * \file PBDNN.hpp
 * Header of the Population of Bi-Directional Neural Networks (PBDNN) class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"
#include "connections/Connection.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/LayerSigmoid.hpp"
#include "../../trainer/errorMeasurers/MSEMeasurer.hpp"

/*!
 * \class PBDNN
 * Population of Bi-Directional Neural Networks (PBDNN).
 */
class PBDNN : public NeuralMachine{
 private :
  std::vector<NeuralNetwork*> forwardPopulation;
  std::vector<FeatureVector> errors;

 protected:

 public:

  /*!
   * Default constructor.
   */
  PBDNN(std::vector<NeuralNetwork*> _forwards);

  /*!
   * Parameter constructor.
   * \param _numNetworks Number of networks to build.
   * \param _numEntries Number of entries for the networks.
   * \param _hiddenLayerSize Number of hidden neurons.
   * \param _mean Mean vector of the dataset.
   * \param _stdev Standard deviation of the dataset.
   */
   PBDNN(uint _numNetworks, uint _numEntries, uint _hiddenLayerSize, ValueVector _mean, ValueVector _stdDev);
  
  /*!
   * Forward a sequence.
   * \param _sequence Sequence to forward.
   */
  virtual void forwardSequence(std::vector<FeatureVector> _sequence);
  
  /*!
   * Get the neural population.
   * \return The population.
   */
  std::vector<NeuralNetwork*> getPopulation() const;

  /*!
   * Get result of a forward propagation.
   * \return The population output.
   */
  std::vector<FeatureVector> getOutputSequence();

  /*!
   * Destructor.
   */
  ~PBDNN();

};


#endif
