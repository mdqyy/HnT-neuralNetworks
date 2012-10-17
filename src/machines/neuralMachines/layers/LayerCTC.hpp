#ifndef __LAYERCTC_HPP__
#define __LAYERCTC_HPP__
/*!
 * \file LayerCTC.hpp
 * Header of the LayerCTC class.
 * \author Luc Mioulet
 */

#include "Layer.hpp"
#include <stdexcept>

/*!
 * \class LayerCTC
 * Hyperbolic tangent layer.
 */
class LayerCTC: public Layer {
private:

protected:

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
	 * Forward a feature vector.
	 */
	void forward();

	/*!
	 * Forward a feature vector.
	 * \param _signal Input signal.
	 * \return Output feature vector.
	 */
	void forward(FeatureVector _signal);

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
