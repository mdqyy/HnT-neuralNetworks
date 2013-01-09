#ifndef CERMEASURER_HPP_
#define CERMEASURER_HPP_
/*!
 * \file CERMeasurer.hpp
 * Header of the CERMeasurer class. Character error rate between a target and sample sequence.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <stdexcept>
#include <iostream>

class CERMeasurer: public ErrorMeasurer {
public:
	/*!
	 * Default constructor.
	 */
	CERMeasurer();

	/*!
	 * Process all errors.
	 * \param _output Machine output feature Vector.
	 * \param _target Dataset result to match.
	 */
	void processErrors(FeatureVector _output, FeatureVector _target);

	/*!
	 * Measure error per unit.
	 * \param _output Machine output feature Vector.
	 * \param _target Dataset result to match.
	 * \return The error vector. Has the same length as the output/target.
	 */
	ErrorVector errorPerUnit(FeatureVector _output, FeatureVector _target);

	/*!
	 * Measure total mean square error of all units.
	 * \param _output Machine output feature Vector.
	 * \param _target Dataset result to match.
	 * \return The total error.
	 */
	realv totalError(FeatureVector _output, FeatureVector _target);

	/*!
	 * Destructor.
	 */
	virtual ~CERMeasurer();
};

#endif /* CERMEASURER_HPP_ */
