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
#include "../trainer/errorMeasurers/ErrorMeasurer.hpp"
#include "PerformanceMeasurer.hpp"

/*!
 * \class RegressionMeasurer
 * Description
 */
class RegressionMeasurer: public PerformanceMeasurer {
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

	/*! Global error */
	realv totalError;

	/*! Percentage of elements */
	realv percentageOfElements;
public:

	/*!
	 * Default constructor.
	 * \param _machine Machine performance to be measured.
	 * \param _data Data to use for measurements.
	 * \param _em Error measurer.
	 */
	RegressionMeasurer(NeuralNetwork& _machine, RegressionDataset& _data, ErrorMeasurer& _em, realv _percentageOfElements = 0.05);

	/*!
	 * Used to define the index order call of the different sequences during validation.
	 * \param _numSequences Number of sequences in the data.
	 * \return Vector of unsigned integers.
	 */
	std::vector<uint> defineIndexOrderSelection(uint _numSequences);

	/*!
	 * Process global mean output error.
	 */
	void processGlobalMeanOutputError();

	/*!
	 * Process mean and standard deviation of output error.
	 */
	void processMeanOutputAndStdDevOutputError();

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

	/*!
	 * Get dataset.
	 * \return Regression dataset.
	 */
	RegressionDataset& getData() const;

	/*!
	 * Get error measurer.
	 * \return Error measurer.
	 */
	ErrorMeasurer& getErrorMeasurer() const;

	/*!
	 * Get machine measured.
	 * \return Machine.
	 */
	Machine& getMachine() const;

	/*!
	 * Mean output error.
	 * \return Feature vector.
	 */
	FeatureVector getMeanOutputError() const;

	/*!
	 * Set mean output error.
	 * \param _meanOutputError Mean output error.
	 */
	void setMeanOutputError(FeatureVector _meanOutputError);

	/*!
	 * Get standard deviation output error.
	 * \return Feature vector.
	 */
	FeatureVector getStdDevOutputError() const;

	/*!
	 * Set standard deviation output error.
	 * \param _stdDevOutputError Standard deviation output error.
	 */
	void setStdDevOutputError(FeatureVector _stdDevOutputError);

	/*!
	 * Get total error.
	 * \return Total error.
	 */
	realv getTotalError() const;

	/*!
	 * Set the total error.
	 * \param _totalError The total error.
	 */
	void setTotalError(realv _totalError);

};

#endif
