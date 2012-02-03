#ifndef __VALUEVECTOR_HPP__
#define __VALUEVECTOR_HPP__
/*!
 * \file ValueVector.hpp
 * Header of the ValueVector class.
 * \author Luc Mioulet
 */

#include "../General.hpp"
#include <iostream>
#include <stdexcept>
#include <opencv/cv.h>

/*!
 * \class ValueVector
 * Description
 */
class ValueVector {
 private :

 protected:
  /*! Matrix containing the data */
  cv::Mat data; 

 public:


  /*!
   * Parameter constructor.
   * \param _length Vector length.
   */
  ValueVector(int _length=1);
  

  /*!
   * Parameter constructor.
   * \param _data Data of the value vector.
   */
  ValueVector(cv::Mat _data);

  /*!
   * Get value vector length.
   * \return Value vector length.
   */
  uint getLength() const;

  /*!
   * Get an element in the matrix
   * \param i Index of the element
   * \return Access to the value vector element.
   */
  const realv& operator[](int i) const;


  /*!
   * Get access to an element in the matrix
   * \param i Index of the element
   * \return Access to the value vector element.
   */
  realv& operator[](int i);

  /*!
   * Reset all values to a default one.
   * \param _default New default value.
   */
  void reset(realv _default);

  /*!
   * Destructor.
   */
  ~ValueVector();

  /*!
   * Output value vector data.
   * \param os Output stream.
   * \param fv Value vector.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const ValueVector& fv);

};


#endif
