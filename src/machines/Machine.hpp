#ifndef __MACHINE_HPP__
#define __MACHINE_HPP__
/*!
 * \file Machine.hpp
 * Header of the Machine class.
 * \author Luc Mioulet
 */

#include "../dataset/FeatureVector.hpp"
#include <string>
#include <vector>

/*!
 * \class Machine
 * Description
 */
class Machine {
 private :

 protected:
  /*! Machine name */
  std::string name;

 public:

  /*!
   * Default constructor.
   */
  Machine();

  /*!
   * Constructor.
   * \param _name Machine name. 
   */
  Machine(std::string _name);

  /*!
   * Get name.
   * \return Machine name.
   */
  std::string getName() const;

  /*!
   * Set name
   * \param _name New name.
   */
  void setName(std::string _name);

  /*!
   * Forward a sequence.
   * \param _sequence Sequence to pass forward.
   */
  virtual void forwardSequence(std::vector<FeatureVector> _sequence)=0;

  /*!
   * Destructor.
   */
  ~Machine();

};


#endif
