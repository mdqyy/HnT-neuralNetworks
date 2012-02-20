#ifndef __NEURALMACHINE_HPP__
#define __NEURALMACHINE_HPP__
/*!
 * \file NeuralMachine.hpp
 * Header of the NeuralMachine class.
 * \author Luc Mioulet
 */

#include "../Machine.hpp"

/*!
 * \class NeuralMachine
 * Description
 */
class NeuralMachine : public Machine {
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  NeuralMachine();

  /*!
   * Parameter constructor.
   */
  NeuralMachine(std::string _name);

  /*!
   * Forward a sequence.
   * \param _sequence Sequence to pass forward.
   */
  virtual void forwardSequence(std::vector<FeatureVector> _sequence)=0;

  /*!
   * Destructor.
   */
  ~NeuralMachine();

};


#endif
