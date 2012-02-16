#ifndef __DATASET_HPP__
#define __DATASET_HPP__
/*!
 * \file Dataset.hpp
 * Header of the Dataset class.
 * \author Luc Mioulet
 */

#include <string>
#include <vector>
#include <stdexcept>
#include "FeatureVector.hpp"
#include <opencv/cv.h>

#define DS_REGRESSION 1
#define DS_CLASSIFICATION 2
#define DS_UNSUPERVISED 3

/*!
 * \class Dataset
 * Dataset class. 
 * Dataset is represented as a vector of vectors of feature vectors. This is done for sequence manipulation.
 */
class Dataset {
 private :

 protected:
  /*! Dataset name */
  std::string name;
  /*! Data vector : a vector of vectors of vectors (dataset-sequence-feature vector). */
  std::vector< std::vector<FeatureVector> > data;
  /*! Number of samples. */
  uint numSamples;
  /*! Number of sequences. */
  uint numSequences;
  /*! Maximum sequence length. If it is 1 the data is not sequential. */
  uint maxSequenceLength;
  /*! Feature vector length*/
  uint fvLength;
  /*! Mean values of the dataset. */
  ValueVector meanMat;
  /*! Matrix used for calculating the standard  deviation of the dataset.*/
  ValueVector Qmat;

 public:

  /*!
   * Default constructor.
   */
  Dataset();

  /*!
   * Get name.
   * \return Dataset name.
   */
  std::string getName() const;

  /*!
   * Get data.
   * \attention Should not be used for very large datasets.
   * \return Data.
   */
  std::vector< std::vector<FeatureVector> > getData() const;

  /*!
   * Get the total number of samples.
   * \return Total number of samples.
   */
  uint getNumSamples() const;

  /*!
   * Get the total number of sequences.
   * \return Total number of sequences.
   */
  uint getNumSequences() const;

  /*!
   * Get maximum sequence length.
   * \return The maximum sequence length.
   */
  uint getMaxSequenceLength() const;

  /*!
   * Get feature vector length.
   * \return The sample feature vector length.
   */
  uint getFeatureVectorLength() const;

  /*!
   * Get the mean vector.
   * \return The mean vector.
   */
  ValueVector getMean() const;

  /*!
   * Get standard deviation.
   * \remark Requires calculations from Qmatrix.
   * \return Stardard deviation.
   */
  ValueVector getStandardDeviation() const;

  /*!
   * Get dataset type.
   * \return Dataset type.
   */
  virtual int getDatasetType() const = 0 ;

  /*! 
   * Give a new name to the data set.
   * \param _name New name of the dataset.
   */
  void setName(std::string _name);
  
  /*!
   * Get sequence at index. 
   * \param _index Data index.
   * \return Vector of feature vector.
   */
  std::vector<FeatureVector>& operator[](uint _index);

  /*!
   * Get sequence at index. 
   * \param _index Data index.
   * \return Vector of feature vector.
   */
  const std::vector<FeatureVector>& operator[](uint _index) const;

  /*! 
   * Update statistics of the database after adding a sample.
   * \param sample Added sample.
   */
  void updateStatistics(FeatureVector _sample);

  /*!
   * Update statistics of the database after adding a sequence.
   * \param _sequence Added sequence.
   */
  void updateStatistics(std::vector<FeatureVector> _sequence);
  
  /*!
   * Load a database from a file.
   * \param _fileName
   */ 
  virtual void load(std::string _fileName)=0;

  /*!
   * Save a database to a file.
   * \param _fileName
   */ 
  virtual void save(std::string _fileName)=0;
  
  /*!
   * Destructor.
   */
  ~Dataset();

};


#endif
