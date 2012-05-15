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
  std::vector<NeuralNetworkPtr> forwardPopulation;
  std::vector<FeatureVector> errors;

 protected:

 public:

  /*!
   * Default constructor.
   */
  PBDNN();

  /*!
   * Parameter constructor.
   * \param _population Population of networks.
   */
  PBDNN(std::vector<NeuralNetworkPtr> _population);

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
  std::vector<NeuralNetworkPtr> getPopulation() const;

  /*!
   * Get result of a forward propagation.
   * \return The population output.
   */
  std::vector<FeatureVector> getOutputSequence();

  /*!
   * Print data concerning the object.
   * \param _os Output file stream.
   */
  void print(std::ostream& _os) const;

  /*!
   * Destructor.
   */
  ~PBDNN();

 /*!
   * Output file stream.
   * \param ofs Output file stream.
   * \param c Connection.
   * \return Output file stream.
   */
  friend std::ofstream& operator<<(std::ofstream& _ofs, const PBDNN& _pop);

  /*!
   * Input file stream.
   * \param _ifs Input file stream.
   * \param _pop Population of neural networks.
   * \return Input file stream.
   */
  friend std::ifstream& operator>>(std::ifstream& _ifs, PBDNN& _pop);

};


#endif
