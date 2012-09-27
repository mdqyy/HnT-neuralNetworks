#ifndef __CLASSIFICATIONERRORMEASURER_HPP__
#define __CLASSIFICATIONERRORMEASURER_HPP__
/*!
 * \file ClassificationErrorMeasurer.hpp
 * Header of the ClassificationErrorMeasurer class.
 * \author Luc Mioulet
 */

#include "ErrorMeasurer.hpp"
#include <opencv/cv.h>

/*!
 * \class ClassificationErrorMeasurer
 * Measure classification errors between a machine output vector and the data target.
 */
class ClassificationErrorMeasurer: ErrorMeasurer {
private:
	uint numberOfExamples;

protected:

public:

	/*!
	 * Default constructor.
	 */
	ClassificationErrorMeasurer();

	/*!
	 * Process all errors.
	 * \param _ouput Machine output feature Vector.
	 * \param _target Dataset result to match.
	 */
	void processErrors(FeatureVector _output, FeatureVector _target);

	/*!
	 * Measure error per unit.
	 * \param ouput Machine output feature Vector.
	 * \param target Dataset result to match.
	 * \return The error vector. Has the same length as the output/target.
	 */
	ErrorVector errorPerUnit(FeatureVector _output, FeatureVector _target);

	/*!
	 * Measure error per unit.
	 * \param ouput Machine output feature Vector.
	 * \param target Dataset result to match.
	 * \return The total error.
	 */
	realv totalError(FeatureVector _output, FeatureVector _target);

	/*!
	 * Destructor.
	 */
	~ClassificationErrorMeasurer();

};

#endif
