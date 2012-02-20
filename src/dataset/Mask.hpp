#ifndef __MASK_HPP__
#define __MASK_HPP__
/*!
 * \file Mask.hpp
 * Header of the Mask class.
 * \author Luc Mioulet
 */

#include <vector>
#include <opencv/cv.h>

/*!
 * \class Mask
 * Description
 */
class Mask {
 private :

 protected:
  std::vector<bool> mask;

 public:

  /*!
   * Parameter constructor.
   * \param _length Mask length.
   */
  Mask(uint _length=0);

  /*!
   * Parameter constructor.
   * \param _mask Matrice mask.
   */
  Mask(cv::Mat _mask);

  /*!
   * Get length of the mask.
   * \return Mask length.
   */
  uint getLength();

  /*!
   * Access index operator.
   * \param _index Index from which information is retrieved.
   */
  const bool& operator[](uint _index) const;

  /*!
   * Access index operator.
   * \param _index Index from which information is retrieved.
   */
  bool operator[](uint _index);

  /*!
   * Reset a mask to a default value.
   * \param _default Default value. 
   */
  void reset(bool _default=true);

  /*!
   * Destructor.
   */
  ~Mask();

};


#endif
