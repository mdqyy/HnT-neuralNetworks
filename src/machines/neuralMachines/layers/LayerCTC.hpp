#ifndef __LAYERCTC_HPP__
#define __LAYERCTC_HPP__
/*!
 * \file LayerCTC.hpp
 * Header of the LayerCTC class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"
#include <map>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <ostream>
#include <stdio.h>

/*!
 * \class LayerCTC
 * Hyperbolic tangent layer.
 */
class LayerCTC: public Layer {
private:

protected:
	/*! Map containing the correspondence between ints and strings representing classes. */
	std::map<int, std::string> classLabels;

	/*! Reversed map containing correspondence between ints and strings representing classes. */
	std::map<std::string, int> classLabelIndex;

public:

	LayerCTC();

	/*!
	 * Parameter constructor.
	 * Blank unit will always be the last unit in the layer.
	 * \param _numUnits Number of units in the layer.
	 * \param _name Name of the layer.
	 */
	LayerCTC(uint _numUnits, std::string _name = "CTC_layer");

	/*!
	 * Copy constructor.
	 * \param _clsm Layer to copy.
	 */
	LayerCTC(const LayerCTC& _clsm);

	/*!
	 * Clone a layer
	 * \return Pointer to a clone.
	 */
	virtual LayerCTC* clone() const;

	/*!
	 * Get the layer type.
	 * \return Layer type.
	 */
	int getLayerType() const;

	/*!
	 * Get class label mapping.
	 * \return Int to label map.
	 */
	std::map<int, std::string> getClassLabelMap();

	/*!
	 * Set class label mapping.
	 * \param _intToString Int to label map.
	 */
	void setClassLabels(std::map<int, std::string> _intToSring);

	/*!
	 * Get class integer mapping.
	 * \return Label to int map.
	 */
	std::map<std::string, int> getClassLabelIndexMap();

	/*!
	 * Set class integer mapping.
	 * \param _stringToInt Label to int map.
	 */
	void setClassLabelIndex(std::map<std::string, int> _stringToInt);


	/*!
	 * Forward a feature vector.
	 */
	void forward();

	/*!
	 * Forward a feature vector.
	 * \param _signal Input signal.
	 * \return Output feature vector.
	 */
	void forward(FeatureVector _signal);

	FeatureVector createInputSignal();

	/*!
	 * Get the result sequence as the maximum outputs neuron of each input sample.
	 * \return The result sequence.
	 */
	std::vector<int> processResultSequence();

	/*!
	 * Get the cleaned result sequence (minus blanks and same letters).
	 * \return The cleaned result sequence.
	 */
	std::vector<int> processCleanedResultSequence();

	/*!
	 * Get the output word from the last input sequence.
	 * \return A word.
	 */
	std::string outputWord();

	/*!
	 * Process the derivative for the layer.
	 * \return A value vector containing the derivative.
	 * \remark Does nothing for CTC, you should use the getDerivatives(std::vector<FeatureVector> _forwardVariables, std::vector<FeatureVector> _backwardVariables) function.
	 */
	ValueVector getDerivatives() const;

	/*!
	 * Process the derivative for the layer.
	 * \return A vector containing the derivatives for all times of a signal.
	 */
	std::vector<ValueVector> getDerivatives(std::vector<FeatureVector> _forwardVariables, std::vector<FeatureVector> _backwardVariables) const;

	/*!
	 * Destructor.
	 */
	virtual ~LayerCTC();

	/*!
	 * Output information to a stream.
	 */
	void print(std::ostream& _os) const;

	/*!
	 * File output stream.
	 * \param ofs Output file stream.
	 * \param l CTC layer.
	 * \return File Output stream.
	 */
	friend std::ofstream& operator<<(std::ofstream& ofs, const LayerCTC& l);

	/*!
	 * File input stream.
	 * \param ifs Input file stream.
	 * \param l CTC layer.
	 * \return File Input stream.
	 */
	friend std::ifstream& operator>>(std::ifstream& _ifs, LayerCTC& _l);


};

#endif
