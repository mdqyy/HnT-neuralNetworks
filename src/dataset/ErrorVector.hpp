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
   * Concatenate a vector of vectors.
   * \param _vecs Vector of vectors.
   */
  ErrorVector(std::vector<ErrorVector> _vecs);

  /*!
   * Destructor.
   */
  ~ErrorVector();

  /*!
   * Output error vector data.
   * \param os Output stream.
   * \param ev Error vector.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, const ErrorVector& ev);

  /*!
   * File output stream.
   * \param ofs Output file stream.
   * \param l Input layer.
   * \return File Output stream.
   */
  friend std::ofstream& operator<<(std::ofstream& ofs, const ErrorVector& ev);

  /*!
   * File input stream.
   * \param ifs Input file stream.
   * \param ev Error vector.
   * \return File Input stream.
   */
  friend std::ifstream& operator>>(std::ifstream& ifs, ErrorVector& ev);

};


#endif
