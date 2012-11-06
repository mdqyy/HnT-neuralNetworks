#ifndef __REGRESSIONDATASET_HPP__
#define __REGRESSIONDATASET_HPP__
/*!
 * \file RegressionDataset.hpp
 * Header of the RegressionDataset class.
 * \author Luc Mioulet
 */

#include "SupervisedDataset.hpp"
#include <string>
#include "../../tinyxml/tinyxml.h"
#include <opencv/cv.h>

/*!
 * \class RegressionDataset
 * Description
 */
class RegressionDataset : public SupervisedDataset{
 private :
  std::vector< std::vector<FeatureVector> > values;

 protected:

 public:

  /*!
   * Default constructor.
   */
  RegressionDataset();

  /*!
   * Add a sequence to the dataset.
   * \param _sequence Sequence data.
   * \param _value  Value vector(s) of the data.
   */
  void addSequence(std::vector<FeatureVector> _sequence, std::vector<FeatureVector> _value);

  /*! 
   * Get the target feature vector for a sequence.
   * \param _i Sequence index.
   * \return The target sequence.
   */
  std::vector<FeatureVector> getTargetSequence(uint _i) const;

  /*! 
   * Get the target feature vector for a sequence.
   * \param _i Sequence index.
   * \param _j Sample in sequence index.
   * \return The target sample.
   */
  FeatureVector getTargetSample(uint _i, uint _j) const;

  /*!
   * Get dataset type.
   * \return Dataset type.
   */
  int getDatasetType() const;

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
  ~RegressionDataset();

  /*!
   * Output stream for the dataset.
   * \param _os Output stream.
   * \param _rd Regression dataset.
   * \return Output stream.
   */
  friend  std::ostream& operator<<(std::ostream& _os, RegressionDataset& _rd);
};


#endif
