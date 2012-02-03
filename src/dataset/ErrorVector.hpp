#ifndef __ERRORVECTOR_HPP__
#define __ERRORVECTOR_HPP__
/*!
 * \file ErrorVector.hpp
 * Header of the ErrorVector class.
 * \author Luc Mioulet
 */

#include "ValueVector.hpp"


/*!
 * \class ErrorVector
 * Contains the error vector produced by Machines.
 */
class ErrorVector : public ValueVector{
 private :

 protected:

 public:


  /*!
   * Parameter constructor.
   * \param _length Vector length.
   */
  ErrorVector(int _length=1);
  

  /*!
   * Parameter constructor.
   * \param _data Data of the error vector.
   */
  ErrorVector(cv::Mat _data);

  /*!
   * Destructor.
   */
  ~ErrorVector();

  /*!
   * Output error vector data.
   * \param os Output stream.
   * \param fv Error vector.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const ErrorVector& fv);

};


#endif
