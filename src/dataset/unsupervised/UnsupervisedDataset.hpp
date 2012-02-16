#ifndef __UNSUPERVISEDDATASET_HPP__
#define __UNSUPERVISEDDATASET_HPP__
/*!
 * \file UnsupervisedDataset.hpp
 * Header of the UnsupervisedDataset class.
 * \author Luc Mioulet
 */

#include "../Dataset.hpp"
#include <string>

/*!
 * \class UnsupervisedDataset
 * Description
 */
class UnsupervisedDataset : public Dataset {
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  UnsupervisedDataset();
 
 /*!
   * Add a sequence.
   * \param sequence Add a complete sequence.
   */
  void addSequence(std::vector<FeatureVector> sequence);

  /*!
   * Add a sample to the last sequence or to the indicated index.
   * \param sample A feature vector.
   * \param index Sequence index.
   */
  void addSample(FeatureVector sample, int index=-1);

  /*!
   * Destructor.
   */
  ~UnsupervisedDataset();

};


#endif
