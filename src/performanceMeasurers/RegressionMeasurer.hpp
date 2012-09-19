#ifndef __REGRESSIONMEASURER_HPP__
#define __REGRESSIONMEASURER_HPP__
/*!
 * \file RegressionMeasurer.hpp
 * Header of the RegressionMeasurer class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <vector>
#include "../machines/neuralMachines/NeuralNetwork.hpp"
#include "../dataset/supervised/RegressionDataset.hpp"
#include "PerformanceMeasurer.hpp"
#include "../trainer/errorMeasurers/ErrorMeasurer.hpp"

/*!
 * \class RegressionMeasurer
 * Description
 */
class RegressionMeasurer : public PerformanceMeasurer {
private:

protected:
	/*! Regression machine */
	NeuralNetwork& machine;

	/*! Regression dataset. */
	RegressionDataset& data;

	/*! Error measurer. */
	ErrorMeasurer& errorMeasurer;

	/*! Mean output error by unit*/
	FeatureVector meanOutputError;

	/*! Standard deviation of output error per unit*/
	FeatureVector stdDevOutputError;
public:

	/*!
	 * Default constructor.
	 * \param _machine Machine performance to be measured.
	 * \param _data Data to use for measurements.
	 * \param _em Error measurer.
	 */
	RegressionMeasurer(NeuralNetwork& _machine, RegressionDataset& _data, ErrorMeasurer& _em);

	/*!
	 * Process global mean output error.
	 */
	realv processGlobalMeanOutputError();

	/*!
	 * Measure performance of a machine.
	 */
	void measurePerformance();

	/*!
	 * Initialize matrices.
	 */
	void initMatrices();

	/*!
	 * Process mean output error.
	 */
	void processMeanOutputError();

	/*!
	 * Destructor.
	 */
	~RegressionMeasurer();
	RegressionDataset& getData() const;
	ErrorMeasurer& getErrorMeasurer() const;
	Machine& getMachine() const;

};

#endif
