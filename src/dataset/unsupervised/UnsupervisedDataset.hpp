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
   int getDatasetType() const;
 
 /*!
   * Add a sequence.
   * \param _sequence Add a complete sequence.
   */
  void addSequence(std::vector<FeatureVector> _sequence);

  /*!
   * Add a sample to the last sequence or to the indicated index.
   * \param _sample A feature vector.
   * \param _index Sequence index.
   */
  void addSample(FeatureVector _sample, int _index=-1);

 /*!
   * Load a database from a file.
   * \param _fileName
   */ 
  void load(std::string _fileName);

  /*!
   * Save a database to a file.
   * \param _fileName
   */ 
  void save(std::string _fileName);

  /*!
   * Destructor.
   */
  ~UnsupervisedDataset();

  /*!
   * Output stream for unsupervised dataset.
   * \param _os Output stream.
   * \param _ud Unsupervised dataset.
   * \return The output stream.
   */
  friend  std::ostream& operator<<(std::ostream& _os, UnsupervisedDataset& _ud);
};


#endif
