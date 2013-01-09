#ifndef __SAMPLEMEASURABLE_HPP__
#define __SAMPLEMEASURABLE_HPP__
/*!
 * \file SampleMeasurer.hpp
 * Header of the Sample Measurable class.
 * \author Luc Mioulet
 */

#include "../../General.hpp"
#include "../../dataset/FeatureVector.hpp"
#include "../../dataset/ErrorVector.hpp"

/*!
 * \class SampleMeasurer
 * Interface for measuring errors on samples.
 */

class SampleMeasurable {
private:

protected:

public:

	/*!
	 * Get error per unit.
	 * \return Error per unit vector.
	 */
	virtual ErrorVector getErrorPerUnit()=0;

	/*!
	 * Set new values for the error per units.
	 * \param _errPerUnit New error per units.
	 */
	virtual void setErrorPerUnit(ErrorVector _errPerUnit)=0;

	/*!
	 * Process all errors.
	 * \param _ouput Machine output feature Vector.
	 * \param _target Dataset result to match.
	 */
	virtual void processErrors(FeatureVector _output, FeatureVector _target)=0;

	/*!
	 * Measure error per unit.
	 * \param _ouput Machine output feature Vector.
	 * \param _target Dataset result to match.
	 * \return The error vector. Has the same length as the output/target.
	 */
	virtual ErrorVector errorPerUnit(FeatureVector _output, FeatureVector _target) = 0;

	/*!
	 * Measure error per unit.
	 * \param ouput Machine output feature Vector.
	 * \param target Dataset result to match.
	 * \return The total error.
	 */
	virtual realv totalError(FeatureVector _output, FeatureVector _target) = 0;

	/*!
	 * Destructor.
	 */
	virtual ~SampleMeasurable() {

	}

};

#endif
