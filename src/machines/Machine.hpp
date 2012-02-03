#ifndef __MACHINE_HPP__
#define __MACHINE_HPP__
/*!
 * \file Machine.hpp
 * Header of the Machine class.
 * \author Luc Mioulet
 */

#include <string>

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
   * Destructor.
   */
  ~Machine();

};


#endif
