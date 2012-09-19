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
public:

	DiversityMeasurer(PBDNN& _population, RegressionDataset& _data);

	void measurePerformance();

	void processNetworkOutputMeanAndStdDevMatrix();

	void processCorrelationMatrix();

	void processCovarianceMatrix();

	void processChiSquareMatrix();

	void processDisagreementMatrix();

	void initMatrices();

	std::vector<cv::Mat> getChiSquareMatrix() const;

	void setChiSquareMatrix(std::vector<cv::Mat> chiSquareMatrix);

	std::vector<cv::Mat> getCorrelationMatrix() const;

	void setCorrelationMatrix(std::vector<cv::Mat> correlationMatrix);

	std::vector<cv::Mat> getCovarianceMatrix() const;

	void setCovarianceMatrix(std::vector<cv::Mat> covarianceMatrix);

	cv::Mat getDisagreementMatrix() const;

	void setDisagreementMatrix(cv::Mat disagreementMatrix);

	RegressionDataset& getData() const;

	void setData(RegressionDataset& data);

	cv::Mat getNetworkOutputMeanMatrix() const;

	void setNetworkOutputMeanMatrix(cv::Mat networkOutputMeanMatrix);

	PBDNN& getNetworkPopulation() const;

	void setNetworkPopulation(PBDNN& networkPopulation);

	cv::Mat getNetworkOutputStdDevMatrix() const;

	void setNetworkOutputStdDevMatrix(cv::Mat stdDevMatrix);

	~DiversityMeasurer();
};

#endif /* DIVERSITYMEASURER_HPP_ */
