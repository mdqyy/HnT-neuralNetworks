#ifndef __UNSUPERVISEDDATASET_HPP__
#define __UNSUPERVISEDDATASET_HPP__
/*!
 * \file UnsupervisedDataset.hpp
 * Header of the UnsupervisedDataset class.
 * \author Luc Mioulet
 */

#include "../Dataset.hpp"
#include <string>
#include "../../tinyxml/tinyxml.h"
#include <opencv/cv.h>

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
   * Get dataset type.
   * \return Dataset type.
   */
  virtual int getDatasetType() const;
 
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
   * Load a database from a file.
   * \param fileName
   */ 
  virtual void load(std::string fileName);

  /*!
   * Save a database to a file.
   * \param fileName
   */ 
  virtual void save(std::string fileName);

  /*!
   * Destructor.
   */
  ~UnsupervisedDataset();

  friend  std::ostream& operator<<(std::ostream& os, UnsupervisedDataset& cd);
};


#endif
