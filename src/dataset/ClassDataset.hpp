#ifndef __CLASSDATASET_HPP__
#define __CLASSDATASET_HPP__
/*!
 * \file ClassDataset.hpp
 * Header of the RegressionDataset class.
 * \author Luc Mioulet
 */

#include <string>
#include <vector>
#include <iostream>
#include <opencv/cv.h>
#include "../utilities/ImageFrameExtractor.hpp"

/*!
 * \class ClassDataset
 * Extract image
 */
class ClassDataset {
 private :
  std::vector<uint> classes;

  uint numberOfClasses;
 protected:

 public:

  /*!
   * Default constructor.
   */
  ClassDataset();

  /*!
   * Parameter constructor.
   * \param _classes.
   * \param _numberOfClasses.
   */
  ClassDataset(std::vector< uint> _classes, uint _numberOfClasses);

  /*! Add a new class information.
   * \param _class name.
   */
  void addClass(uint _class);
  
  /*! Create a class Feature Vector.
   * \param _index The index key.
   */
  FeatureVector getFeatureVector(uint _index);

  /*! Get the class at index i
   * \param _index The index key.
   * \return Class at index i.
   */
  uint getClass(uint _index) const;

  /*! Get classes length.
   * \return
   */
  uint getClassesLength() const;

  /*! Get the classes
   * \return The classes.
   */
  std::vector<uint> getClasses() const;

  /*! Set the classes.
   * \param _.
   */
  void setClasses(std::vector<uint> _classes);

  /* Get number of classes.
   * \return Number of classes.
   */
  uint getNumberOfClasses() const;

  /* Set the number of classes.
   * \param _numClasses Number of classes
   */
  void setNumberOfClasses(uint _numClasses);

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
  ~ClassDataset();

  /*!
   * Output stream for the dataset.
   * \param _os Output stream.
   * \param _rd Regression dataset.
   * \return Output stream.
   */
  friend  std::ostream& operator<<(std::ostream& _os, ClassDataset& _cd);
};


#endif
