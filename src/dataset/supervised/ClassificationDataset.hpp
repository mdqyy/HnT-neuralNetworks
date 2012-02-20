#ifndef __CLASSIFICATIONDATASET_HPP__
#define __CLASSIFICATIONDATASET_HPP__
/*!
 * \file ClassificationDataset.hpp
 * Header of the ClassificationDataset class.
 * \author Luc Mioulet
 */

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <opencv/cv.h>
#include "SupervisedDataset.hpp"
#include "../../tinyxml/tinyxml.h"


/*!
 * \class ClassificationDataset
 * Classification dataset for learning tasks.
 */
class ClassificationDataset : public SupervisedDataset{
 private :
  
 protected:
  /*! Map containeng correspondance between ints and strings representing classes. */
  std::map<int , std::string> classLabels;
  /*! Reversed map containing correspondance between ints and strings representing classes. */
  std::map<std::string, int> classLabelIndex;
  /*! Vector of vector of classes. Should be associated to the data in these three ways :
   * - one class per sequence,
   * - same number of classes as of samples in the sequence,
   * - number of classes inferior to the number of samples in the sequence.
   */
  std::vector< std::vector<int> > classes;
  /*! Maximum class vector length. Should not be superior to the maximum sequence length. */
  uint maxClasses;
 public:

  /*!
   * Default constructor.
   */
  ClassificationDataset();

  /*!
   * Get the number of classes.
   * \return The number of classes.
   */
  int getNumberOfClasses();
  
  
  /*!
   * Get class label mapping.
   * \return Int to label map.
   */
  std::map<int, std::string> getClassLabelMap();

  /*!
   * Get class integer mapping.
   * \return Label to int map.
   */
  std::map<std::string, int> getClassLabelIndexMap();

  /*! 
   * Get index from class.
   * \param _class Class.
   * \return The corresponding index of the label.
   */
  int getIndexLabel(std::string _class) const;

  /*!
   * Get class from index.
   * \param _index Index.
   * \return The corresponding label of the index.
   */
  std::string getClassLabel(int _index) const;

  /*!
   * Get classes for sequence index.
   * \param _index Index of class.
   * \return Vector of classes (as strings).
   */
  std::vector<std::string> getSequenceClasses(uint _index) const;

  /*!
   * Get index of classes for a sequence.
   * \param _index Index of class.
   * \return Vector of classes (as ints).
   */
  std::vector<int> getSequenceClassesIndex(uint _index) const;

  /*! 
   * Get the target feature vector for a sequence.
   * \param _i Sequence index.
   * \return The target sequence targets.
   */
  virtual std::vector<FeatureVector> getTargetSequence(uint _i) const;

  /*! 
   * Get the target feature vector for a sequence.
   * \param _i Sequence index.
   * \param _j Sample index in sequence.
   * \return The target sequence targets.
   */
  virtual FeatureVector getTargetSample(uint _i, uint _j) const;
  
  /*!
   * Get sample class index.
   * \param _i Sequence index.
   * \param _j Sample index.
   * \return Class index of the sample.
  */
  int getSampleClassIndex(uint _i,uint _j) const; 

  /*!
   * Get sample class.
   * \param _i Sequence index.
   * \param _j Sample index.
   * \return Class of the sample.
   */
  std::string getSampleClass(uint _i, uint _j) const;

  /*!
   * Get dataset type.
   * \return Dataset type.
   */
  virtual int getDatasetType() const;

  /*!
   * Add a sequence and its label.
   * \param _sequence Add a feature vector as a complete sequence.
   * \param _classes Classes as an int.
   */
  void addSequence(FeatureVector _sequence,int _class);

  /*!
   * Add a sequence and its label.
   * \param _sequence Add a feature vector as a complete sequence.
   * \param _classes Classes as a string.
   */
  void addSequence(FeatureVector _sequence,std::string _class);
 
  /*!
   * Add a sequence and its label.
   * \param _sequence Add a complete sequence.
   * \param _classes Classes vector as a vector of ints.
   */
  void addSequence(std::vector<FeatureVector>& _sequence, std::vector<int>& _classes);

  /*!
   * Add a sequence and its label.
   * \param sequence Add a complete sequence.
   * \param classes Classes vector as a vector of strings.
   */
  void addSequence(std::vector<FeatureVector>& sequence, std::vector<std::string>& classes);

  /*!
   * Add a sample (and perhaps the label) to the last sequence or to the indicated index.
   * \param sample A feature vector.
   * \param classIndex Class index.
   * \param index Sequence index.
   */
  void addSample(FeatureVector& sample, int classIndex = -1, uint index = -1);

  /*!
   * Add a sample (and perhaps the label) to the last sequence or to the indicated index.
   * \param sample A feature vector.
   * \param className Class.
   * \param index Sequence index.
   */
  void addSample(FeatureVector& sample, std::string className = "", uint index = -1);
  
  /*! 
   * Add a class to the class maps.
   * \param _class Name of the class.
   * \param _index Index of the class. If not defined will be added to the map as the follozing point.
   */
  void addClass(std::string _class, int _index=-1);

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
  ~ClassificationDataset();

    /*!
   * Output Classification dataset.
   * \param os Output stream.
   * \param cd Feature vector.
   * \return Output stream.
   */
  friend std::ostream& operator<<(std::ostream& os, ClassificationDataset& cd);

};


#endif
