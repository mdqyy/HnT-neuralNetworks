#ifndef __PBDNN_HPP__
#define __PBDNN_HPP__
/*!
 * \file PBDNN.hpp
 * Header of the Population of Bi-Directional Neural Networks (PBDNN) class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"
#include "../../trainer/errorMeasurers/MSEMeasurer.hpp"

/*!
 * \class PBDNN
 * Population of Bi-Directional Neural Networks (PBDNN).
 */
class PBDNN : public NeuralMachine{
 private :
  std::vector<NeuralNetwork*>& forwardPopulation;
  std::vector<FeatureVector> errors;

 protected:

 public:

  /*!
   * Default constructor.
   */
  PBDNN(std::vector<NeuralNetwork*>& _forwards);
  
  /*!
   * Forward a sequence.
   * \param _sequence Sequence to forward.
   */
  virtual void forwardSequence(std::vector<FeatureVector> _sequence);
  
  /*!
   * Get the neural population.
   * \return The population.
   */
  std::vector<NeuralNetwork*>& getPopulation() const;

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
