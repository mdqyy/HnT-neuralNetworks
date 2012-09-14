#ifndef __PERFORMANCEMEASURER_HPP__
#define __PERFORMANCEMEASURER_HPP__
/*!
 * \file PerformanceMeasurer.hpp
 * Header of the PerformanceMeasurer class.
 * \author Luc Mioulet
 */

/*!
 * \class PerformanceMeasurer
 * Description
 */
class PerformanceMeasurer {
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  PerformanceMeasurer();

  /*!
   * Measure performance of a Machine on a dataset.
   */
  virtual void measurePerformance()=0;

  /*!
   * Destructor.
   */
  virtual ~PerformanceMeasurer();

};


#endif
