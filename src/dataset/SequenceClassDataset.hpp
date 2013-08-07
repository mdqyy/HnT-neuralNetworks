#ifndef __SEQUENCECLASSDATASET_HPP__
#define __SEQUENCECLASSDATASET_HPP__
/*!
 * \file SequenceClassDataset.hpp
 * Header of the SequenceClassDataset class.
 * \author Luc Mioulet
 */

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <opencv/cv.h>
#include "FeatureVector.hpp"

/*!
 * \class SequenceClassDataset
 * Classification dataset for learning tasks.
 */
class SequenceClassDataset{
private:

protected:
	/*! Map containing the correspondence between ints and strings representing classes. */
	std::map<int, std::string> classLabels;

	/*! Reversed map containing correspondence between ints and strings representing classes. */
	std::map<std::string, int> classLabelIndex;

	/*! Vector of vector of classes. Should be associated to the data in these three ways :
	 * - one class per sequence,
	 * - same number of classes as of samples in the sequence,
	 * - number of classes inferior to the number of samples in the sequence.
	 */
	std::vector<std::vector<int> > classes;

	/*! Maximum class vector length. Should not be superior to the maximum sequence length. */
	uint maxClasses;
public:

	/*!
	 * Default constructor.
	 */
	SequenceClassDataset();


	/*!
	 * Parameter constructor.
	 */
	SequenceClassDataset(std::map<int, std::string> _classLabels, std::map<std::string, int> _classLabelIndex, std::vector<std::vector<int> > _classes, uint _maxClasses);

	/*!
	 * Get the number of classes.
	 * \return The number of classes.
	 */
	int getNumberOfClasses() const;

	/*!
	 * Get class label mapping.
	 * \return Int to label map.
	 */
	std::map<int, std::string> getClassLabelMap() const;

	/*!
	 * Get class integer mapping.
	 * \return Label to int map.
	 */
	std::map<std::string, int> getClassLabelIndexMap() const;

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
	std::vector<FeatureVector> getTargetSequence(uint _i) const;

	/*!
	 * Get the target feature vector for a sequence.
	 * \param _i Sequence index.
	 * \param _j Sample index in sequence.
	 * \return The target sequence targets.
	 */
	FeatureVector getTargetSample(uint _i, uint _j) const;

	/*!
	 * Get sample class index.
	 * \param _i Sequence index.
	 * \param _j Sample index.
	 * \return Class index of the sample.
	 */
	int getSampleClassIndex(uint _i, uint _j) const;

	/*!
	 * Get sample class.
	 * \param _i Sequence index.
	 * \param _j Sample index.
	 * \return Class of the sample.
	 */
	std::string getSampleClass(uint _i, uint _j) const;

	/*!
	 *  Get classes
	 * \return Vector of vector containing the classes.
	 */
	std::vector<std::vector<int> > getClasses() const;

	/*!
	 * Get the mapping from string to int.
	 * \return Map of strings to int.
	 */
	std::map<std::string, int> getClassLabelIndex() const;

	/*!
	 * Set the mapping from string to int.
	 * \param _classLabelIndex Map of strings to int.
	 */
	void setClassLabelIndex(std::map<std::string, int> _classLabelIndex);

	/*!
	 * Get class map from int to strings.
	 * \return Map of int to strings.
	 */
	std::map<int, std::string> getClassLabels() const;

	/*!
	 * Set class map from int to strings.
	 * \param _classLabels Map of int to strings.
	 */
	void setClassLabels(std::map<int, std::string> _classLabels);

	/*!
	 * Get maximum length of a class.
	 * \param Maximum class length.
	 */
	uint getMaxClasses() const;

	/*!
	 * Add a sequence and its label.
	 * \param _class Class as an int.
	 */
	void addSequence(int _class);

	/*!
	 * Add a sequence and its label.
	 * \param _class Class as a string.
	 */
	void addSequence(std::string _class);

	/*!
	 * Add a sequence and its label.
	 * \param _classes Classes vector as a vector of ints.
	 */
	void addSequence(std::vector<int> _classes);

	/*!
	 * Add a sequence and its label.
	 * \param _classes Classes vector as a vector of strings.
	 */
	void addSequence(std::vector<std::string> _classes);

	/*!
	 * Add a sample (and perhaps the label) to the last sequence or to the indicated index.
	 * \param _classIndex Class index.
	 * \param _index Sequence index.
	 */
	void addSample(int _classIndex = -1, uint _index = -1);

	/*!
	 * Add a sample (and perhaps the label) to the last sequence or to the indicated index.
	 * \param _className Class.
	 * \param _index Sequence index.
	 */
	void addSample(std::string _className = "", uint _index = -1);

	/*!
	 * Add a class to the class maps.
	 * \param _class Name of the class.
	 * \param _index Index of the class. If not defined will be added to the map as the following point.
	 */
	void addClass(std::string _class, int _index = -1);

	/*!
	 * Load a database from a file.
	 * \param fileName
	 */
	void load(std::string _fileName);

	/*!
	 * Save a database to a file.
	 * \param fileName
	 */
	void save(std::string _fileName);

	/*!
	 * Destructor.
	 */
	~SequenceClassDataset();

	/*!
	 * Output Classification dataset.
	 * \param os Output stream.
	 * \param cd Feature vector.
	 * \return Output stream.
	 */
	friend std::ostream& operator<<(std::ostream& os, SequenceClassDataset& cd);

};

#endif
