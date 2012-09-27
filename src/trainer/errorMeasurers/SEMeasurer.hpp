#ifndef __SEMEASURER_HPP__
#define __SEMEASURER_HPP__
/*!
 * \file SEMeasurer.hpp
 * Header of the SEMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <stdexcept>
#include <iostream>

/*!
 * \class SEMeasurer
 * Measure errors between a machine output vector and the data target.
 */
class SEMeasurer: public ErrorMeasurer {
private:

protected:

public:

	/*!
	 * Default constructor.
	 */
	SEMeasurer();

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
	~SEMeasurer();

	/*!
	 * Output stream operator.
	 * \param _os Output stream.
	 * \param _se Error measurer.
	 * \return Output stream.
	 */
	friend std::ostream& operator<<(std::ostream& _os, SEMeasurer& _se);

};

#endif
