#ifndef __SUPERVISEDDATASET_HPP__
#define __SUPERVISEDDATASET_HPP__
/*!
 * \file SupervisedDataset.hpp
 * Header of the SupervisedDataset class.
 * \author Luc Mioulet
 */

#include "../Dataset.hpp"
#include <string>

/*!
 * \class SupervisedDataset
 * Description
 */
class SupervisedDataset : public Dataset{
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  SupervisedDataset();

  /*! 
   * Get the target feature vector for a sequence.
   * \param _i Sequence index.
   * \return The target sequence targets.
   */
  virtual std::vector<FeatureVector> getTargetSequence(uint _i) const = 0;

  /*! 
   * Get the target feature vector for a sequence.
   * \param _index Sequence index.
   * \return The target sequence targets.
   */
  virtual FeatureVector getTargetSample(uint _i, uint _j) const = 0;
  
  /*!
   * Get dataset type.
   * \return Dataset type.
   */
  virtual int getDatasetType() const = 0 ;
  
  /*!
   * Destructor.
   */
  ~SupervisedDataset();

};


#endif
