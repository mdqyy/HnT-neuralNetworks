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

  /* Todo : Methods*/

  /*!
   * Destructor.
   */
  ~NeuralMachine();

};


#endif
