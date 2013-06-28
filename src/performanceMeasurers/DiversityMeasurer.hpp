#ifndef DIVERSITYMEASURER_HPP_
#define DIVERSITYMEASURER_HPP_

/*!
 * \file DiversityMeasurer.hpp
 * Header of the DiversityMeasurer class.
 * \author Luc Mioulet
 */

/*!
 * \class DiversityMeasurer
 * Measures diversity among a population of networks.
 * Measures used are correlation, covariance, chi-square, disagreement.
 * The first three are 3 dimensional matrices (vector<cv::Mat>) and values should be as small as possible for a big diversity.
 * The last one is a 2 dimensional matrix (cv::Mat) and values close to 1 mean high diversity.
 */
#include <opencv/cv.h>
#include <vector>
#include "../machines/neuralMachines/PBDNN.hpp"
#include "../dataset/supervised/RegressionDataset.hpp"
#include "../trainer/errorMeasurers/ErrorMeasurer.hpp"
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

	/*! Error measurer. */
	ErrorMeasurer& errorMeasurer;

	/*! Percentage of population */
	realv percentageOfPopulation;
public:
	/*!
	 * Parameter constructor.
	 * \param _population Population.
	 * \param _data Dataset To test diversity.
	 */
	DiversityMeasurer(PBDNN& _population, RegressionDataset& _data, ErrorMeasurer& _em, realv _percentageOfPopulation=0.05);

	/*!
	 * Used to define the index order call of the different sequences during validation.
	 * \param _numSequences Number of sequences in the data.
	 * \return Vector of unsigned integers.
	 */
	std::vector<uint> defineIndexOrderSelection(uint _numSequences);

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
	 * Find the best network for every sample in the database.
	 * \return The vector of assignments. Should be bijective to the dataset given.
	 */
	std::vector<std::vector<int> > findBestNetwork();

	/*!
	 * Find the best network error for every sample.
	 * \return The error vector of the networks on only their best samples.
	 */
	std::vector<realv> errorsOnBestSample();

	/*!
	 * Find the best network error for every sample.
	 * \return The number fo samples assigned to each network.
	 */
	std::vector<int> sampleRepartition();

	/*!
	 * Build a composite answer only taking the best network output for every sample in the database.
	 * \return The vector of assignments. Should be bijective to the dataset given.
	 */
	std::vector<std::vector<FeatureVector> > buildBestOutput();

	/*!
	 * Outputs the mean output of every network. Only mean output of besties.
	 * \return The mean output of every network on best samples.
	 */
	std::vector<FeatureVector> getMeanGoodOutput();

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

	/*!
	 * Get the network output mean matrix.
	 * \return Network output mean matrix.
	 */
	cv::Mat getNetworkOutputMeanMatrix() const;

	/*!
	 * Set the network output mean matrix.
	 * \param _networkOutputMeanMatrix Network output mean matrix.
	 */
	void setNetworkOutputMeanMatrix(cv::Mat _networkOutputMeanMatrix);

	/*!
	 * Get the network population.
	 * \return The network population.
	 */
	PBDNN& getNetworkPopulation() const;

	/*!
	 * Set the network population.
	 * \param _networkPopulation The network population.
	 */
	void setNetworkPopulation(PBDNN& _networkPopulation);

	/*!
	 * Get the output standard deviation matrix.
	 * \return The network output standard deviation matrix.
	 */
	cv::Mat getNetworkOutputStdDevMatrix() const;

	/*!
	 * Set the output standard deviation matrix.
	 * \param The network output standard deviation matrix.
	 */
	void setNetworkOutputStdDevMatrix(cv::Mat _stdDevMatrix);

	/*!
	 * Get the disagreement scalar.
	 * \return The disagreement scalar.
	 */
	realv getDisagreementScalar() const;

	/*!
	 * Set the disagreement scalar.
	 * \param The disagreement scalar.
	 */
	void setDisagreementScalar(realv _disagreementScalar);

	/*!
	 * Get the error measurer.
	 * \return The error measurer.
	 */
	ErrorMeasurer& getErrorMeasurer() const;

	/*!
	 * Destructor.
	 */
	~DiversityMeasurer();

};

#endif /* DIVERSITYMEASURER_HPP_ */
