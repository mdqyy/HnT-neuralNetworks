#ifndef __CLASSIFICATIONMEASURER_HPP__
#define __CLASSIFICATIONMEASURER_HPP__
/*!
 * \file ClassificationMeasurer.hpp
 * Header of the ClassificationMeasurer class.
 * \author Luc Mioulet
 */

/*!
 * \class ClassificationMeasurer
 * Description
 */
class ClassificationMeasurer {
 private :
  

 protected:

 public:

  /*!
   * Default constructor.
   */
  ClassificationMeasurer();

  /*!
   * Measure machine results on a dataset.
   * \param _machine Machine.
   * \param _dataset Dataset.
   */
  measure(Machine& _machine, ClassificationDataset& _dataset);

  /*!
   * Destructor.
   */
  ~ClassificationMeasurer();

};


#endif
