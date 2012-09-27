#ifndef DIVERSITYMEASURER_HPP_
#define DIVERSITYMEASURER_HPP_

/*!
 * \file DiversityMeasurer.hpp
 * Header of the DiversityMeasurer class.
 * \author Luc Mioulet
 */

/*!
 * \class DiversityMeasurer
 * Measures diversity among a population of
 */
#include <opencv/cv.h>
#include <vector>
#include "../machines/neuralMachines/PBDNN.hpp"
#include "../dataset/supervised/RegressionDataset.hpp"
#include "PerformanceMeasurer.hpp"

class DiversityMeasurer: public PerformanceMeasurer {
private:
protected:
	/*! Network population */
	PBDNN& networkPopulation;

	/*! Regression dataset to be used to measure diversity. */
	RegressionDataset& data;

	/*! Mean values of the outputs of all networks over all data samples. */
	cv::Mat networkOutputMeanMatrix;

	/*! Standard deviation of the outputs of all networks over all data samples. */
	cv::Mat networkOutputStdDevMatrix;

	/*! Correlation matrix between networks. */
	std::vector<cv::Mat> correlationMatrix;

	/*! Covariance matrix between networks. */
	std::vector<cv::Mat> covarianceMatrix;

	/*! Chi square matrix between networks. */
	std::vector<cv::Mat> chiSquareMatrix;

	/*! Disagreement matrix between networks. */
	cv::Mat disagreementMatrix;

	/*! Disagreement scalar. */
	realv disagreementScalar;
public:
	/*!
	 * Parameter constructor.
	 * \param _population Population.
	 * \param _data Dataset To test diversity.
	 */
	DiversityMeasurer(PBDNN& _population, RegressionDataset& _data);

	/*!
	 * Measure performances.
	 */
	void measurePerformance();

	/*!
	 * Process network output mean and standard deviation matrix.
	 * This process accelerates calculation.
	 */
	void processNetworkOutputMeanAndStdDevMatrix();

	/*!
	 * Process correlation matrix.
	 */
	void processCorrelationMatrix();

	/*!
	 * Process covariance matrix.
	 */
	void processCovarianceMatrix();

	/*!
	 * Process chi square matrix.
	 */
	void processChiSquareMatrix();

	/*!
	 * Process disagreement measure.
	 */
	void processDisagreementMatrix();

	/*!
	 * Process disagreement scalar.
	 */
	void processDisagreementScalar();

	/*!
	 * Initialize matrices
	 */
	void initMatrices();

	/*!
	 * Get chi square matrix.
	 * \return Chi square matrix.
	 */
	std::vector<cv::Mat> getChiSquareMatrix() const;

	/*!
	 * Set chi square matrix.
	 * \param _chiSquareMatrix Chi square matrix.
	 */
	void setChiSquareMatrix(std::vector<cv::Mat> _chiSquareMatrix);

	/*!
	 * Get correlation matrix.
	 * \return Correlation matrix.
	 */
	std::vector<cv::Mat> getCorrelationMatrix() const;

	/*!
	 * Set the corellation matrix.
	 * \param _correlationMatrix Corellation matrix.
	 */
	void setCorrelationMatrix(std::vector<cv::Mat> _correlationMatrix);

	/*!
	 * Get the covariance matrix.
	 * \return Covariance matrix.
	 */
	std::vector<cv::Mat> getCovarianceMatrix() const;

	/*!
	 * Set the covariance matrix.
	 * \param _covarianceMatrix.
	 */
	void setCovarianceMatrix(std::vector<cv::Mat> _covarianceMatrix);

	/*!
	 * Get the disagreement matrix.
	 * \return Disagreement matrix.
	 */
	cv::Mat getDisagreementMatrix() const;

	/*!
	 * Set the disagreement matrix
	 * \param _disagreementMatrix The disagreement matrix.
	 */
	void setDisagreementMatrix(cv::Mat _disagreementMatrix);

	/*!
	 * Get the dataset.
	 * \return The dataset.
	 */
	RegressionDataset& getData() const;

	/*!
	 * Set the dataset.
	 * \param _data The dataset.
	 */
	void setData(RegressionDataset& _data);

	cv::Mat getNetworkOutputMeanMatrix() const;

	void setNetworkOutputMeanMatrix(cv::Mat _networkOutputMeanMatrix);

	PBDNN& getNetworkPopulation() const;

	void setNetworkPopulation(PBDNN& networkPopulation);

	cv::Mat getNetworkOutputStdDevMatrix() const;

	void setNetworkOutputStdDevMatrix(cv::Mat stdDevMatrix);

	realv getDisagreementScalar() const;

	void setDisagreementScalar(realv disagreementScalar);

	~DiversityMeasurer();
};

#endif /* DIVERSITYMEASURER_HPP_ */
