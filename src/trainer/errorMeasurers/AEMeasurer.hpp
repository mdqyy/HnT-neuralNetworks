#ifndef __AEMEASURER_HPP__
#define __AEMEASURER_HPP__
/*!
 * \file AEMeasurer.hpp
 * Header of the AEMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <stdexcept>
#include <iostream>

/*!
 * \class AEMeasurer
 * Measure absolute errors between a machine output vector and the data target.
 */
class AEMeasurer: public ErrorMeasurer {
private:

protected:

public:

	/*!
	 * Default constructor.
	 */
	AEMeasurer();

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
	~AEMeasurer();

	/*!
	 * Output stream operator.
	 * \param _os Output stream.
	 * \param _ae Error measurer.
	 * \return Output stream.
	 */
	friend std::ostream& operator<<(std::ostream& _os, AEMeasurer& _ae);

};

#endif
