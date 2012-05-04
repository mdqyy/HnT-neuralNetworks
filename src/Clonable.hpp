#ifndef __CLONABLE_HPP__
#define __CLONABLE_HPP__
/*!
 * \file Clonable.hpp
 * Header of the Clonable class.
 * \author Luc Mioulet
 */

#include <boost/shared_ptr.hpp>

/*!
 * \class Clonable
 * Description
 */

class Clonable {
 private :

 protected:

 public:

  /*!
   * Clone method interface.
   * \return Pointer to a clone.
   */
  virtual Clonable* clone() const = 0;
  
  /*!
   * Destructor.
   */
  virtual ~Clonable() {}

 
};


#endif
