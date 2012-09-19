/*!
 * \file DiversityMeasurer.cpp
 * Body of the DiversityMeasurer class.
 * \author Luc Mioulet
 */

#include "DiversityMeasurer.hpp"

using namespace cv;
using namespace std;

DiversityMeasurer::DiversityMeasurer(PBDNN& _population, RegressionDataset& _data) : networkPopulation(_population), data(_data)/*, networkOutputMeanMatrix(Mat()), networkOutputStdDevMatrix(Mat()), correlationMatrix(vector<Mat>()), covarianceMatrix(vector<Mat>()), chiSquareMatrix(vector<Mat>()), disagreementMatrix(vector<Mat>())*/ {
	initMatrices();
}

void DiversityMeasurer::measurePerformance() {
	processNetworkOutputMeanAndStdDevMatrix();
	/*processCorrelationMatrix();
	processCovarianceMatrix();
	processChiSquareMatrix();*/
	processDisagreementMatrix();
}

void DiversityMeasurer::initMatrices() {
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector output = population[0]->getOutputSignal();
	networkOutputMeanMatrix = Mat::zeros(output.getLength(), population.size(), CV_64FC1);
	networkOutputStdDevMatrix = Mat::zeros(output.getLength(),	population.size(),CV_64FC1);
	correlationMatrix = vector<Mat>(networkOutputMeanMatrix.rows, Mat::zeros(population.size(), population.size(), CV_64FC1));
	covarianceMatrix = vector<Mat>(networkOutputMeanMatrix.rows, Mat::zeros(population.size(), population.size(), CV_64FC1));
	chiSquareMatrix = vector<Mat>(networkOutputMeanMatrix.rows, Mat::zeros(population.size(), population.size(), CV_64FC1));
	disagreementMatrix = Mat::zeros(population.size(), population.size(), CV_64FC1);
}


void DiversityMeasurer::processNetworkOutputMeanAndStdDevMatrix(){
	FeatureVector networkOutput;
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	Mat test = Mat::zeros(1,1,CV_64FC1);
	double numSamples = 1.0;
	double oldMean = 0.0;
	for (uint i = 0; i < population.size(); i++) {
		for (uint j = 0; j < data.getNumSequences(); j++) {
			for (uint k = 0; k < data[j].size(); k++) {
				population[i]->forward(data[j][k]);
				networkOutput = population[i]->getOutputSignal();
				for (uint l = 0; l < networkOutput.getLength(); l++) {
					oldMean = networkOutputMeanMatrix.at<double>(l, i);
					networkOutputMeanMatrix.at<double>(l, i) = networkOutputMeanMatrix.at<double>(l, i) + (networkOutput[l] -networkOutputMeanMatrix.at<double>(l, i)) / numSamples;
					networkOutputStdDevMatrix.at<double>(l, i) = ((numSamples - 1) * networkOutputStdDevMatrix.at<double>(l, i)	+ (networkOutput[l] - networkOutputMeanMatrix.at<double>(l, i)) * (networkOutput[l] - oldMean)) / numSamples;
					numSamples = numSamples + 1.0;
				}
			}
		}
	}
}

void DiversityMeasurer::processCorrelationMatrix(){
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector outputA, outputB;
	for (uint j = 0; j < population.size(); j++) {
		for (uint k = j + 1; k < population.size(); k++) {
			vector<double> jSum = vector<double>(networkOutputMeanMatrix.rows, 0.0);
			vector<double> kSum = vector<double>(networkOutputMeanMatrix.rows, 0.0);
			for (uint l = 0; l < data.getNumSequences(); l++) {
				for (uint m = 0; m < data[l].size(); m++) {
					population[j]->forward(data[l][m]);
					population[k]->forward(data[l][m]);
					outputA = population[j]->getOutputSignal();
					outputB = population[k]->getOutputSignal();
					for (uint n = 0; n < outputA.getLength(); n++) {
						correlationMatrix[n].at<double>(j, k) = correlationMatrix[n].at<double>(j, k) + (outputA[n] - networkOutputMeanMatrix.at<double>(n, j)) * (outputB[n] - networkOutputMeanMatrix.at<double>(n, k));
						jSum[n] = kSum[n] + (outputA[n] - networkOutputMeanMatrix.at<double>(n, j)) * (outputA[n] - networkOutputMeanMatrix.at<double>(n, j));
						kSum[n] = jSum[n] + (outputB[n] - networkOutputMeanMatrix.at<double>(n, k)) * (outputB[n] - networkOutputMeanMatrix.at<double>(n, k));
					}
				}
			}
			for (uint n = 0; n < outputA.getLength(); n++) {
				correlationMatrix[n].at<double>(j, k) = correlationMatrix[n].at<double>(j, k) / (jSum[n] * kSum[n]);
				correlationMatrix[n].at<double>(k, j) = correlationMatrix[n].at<double>(j, k);
			}
		}
	}
}

void DiversityMeasurer::processCovarianceMatrix(){
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector outputA, outputB;
	for (uint j = 0; j < population.size(); j++) {
		for (uint k = j + 1; k < population.size(); k++) {
			for (uint l = 0; l < data.getNumSequences(); l++) {
				for (uint m = 0; m < data[l].size(); m++) {
					population[j]->forward(data[l][m]);
					population[k]->forward(data[l][m]);
					outputA = population[j]->getOutputSignal();
					outputB = population[k]->getOutputSignal();
					for (uint n = 0; n < outputA.getLength(); n++) {
						covarianceMatrix[n].at<double>(j, k) = covarianceMatrix[n].at<double>(j, k) + (outputA[n] - networkOutputMeanMatrix.at<double>(n, j)) * (outputB[n] - networkOutputMeanMatrix.at<double>(n, k));
					}
				}
			}
			for (uint n = 0; n < outputA.getLength(); n++) {
				covarianceMatrix[n].at<double>(j, k) = covarianceMatrix[n].at<double>(j, k) / ((double)data.getNumSamples());
				covarianceMatrix[n].at<double>(k, j) = covarianceMatrix[n].at<double>(j, k);
			}
		}
	}
}

void DiversityMeasurer::processChiSquareMatrix(){
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector outputA, outputB;
	for (uint j = 0; j < population.size(); j++) {
		for (uint k = j + 1; k < population.size(); k++) {
			for (uint l = 0; l < data.getNumSequences(); l++) {
				for (uint m = 0; m < data[l].size(); m++) {
					population[j]->forward(data[l][m]);
					population[k]->forward(data[l][m]);
					outputA = population[j]->getOutputSignal();
					outputB = population[k]->getOutputSignal();
					for (uint n = 0; n < outputA.getLength(); n++) {
						chiSquareMatrix[n].at<double>(j, k) = chiSquareMatrix[n].at<double>(j, k) + (outputA[n] - outputB[n]) * (outputA[n] - outputB[n])/outputB[n];
						chiSquareMatrix[n].at<double>(k, j) = chiSquareMatrix[n].at<double>(j, k) + (outputA[n] - outputB[n]) * (outputA[n] - outputB[n])/outputA[n];
					}
				}
			}
		}
	}
}

void DiversityMeasurer::processDisagreementMatrix(){
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector outputA, outputB, target;
	double agreement,disagreementA, disagreementB;
	bool AGood, BGood;
	for (uint j = 0; j < population.size(); j++) {
		for (uint k = j + 1; k < population.size(); k++) {
			vector<double> jSum = vector<double>(networkOutputMeanMatrix.rows, 0.0);
			vector<double> kSum = vector<double>(networkOutputMeanMatrix.rows, 0.0);
			agreement = 0.0;
			disagreementA = 0.0;
			disagreementB = 0.0;
			for (uint l = 0; l < data.getNumSequences(); l++) {
				for (uint m = 0; m < data[l].size(); m++) {
					population[j]->forward(data[l][m]);
					population[k]->forward(data[l][m]);
					outputA = population[j]->getOutputSignal();
					outputB = population[k]->getOutputSignal();
					target = data.getTargetSample(l,m);
					for (uint n = 0; n < outputA.getLength(); n++) {
						if(outputA[n] > target[n] - sqrt(networkOutputStdDevMatrix.at<double>(j,n)) && outputA[n] < target[n] + sqrt(networkOutputStdDevMatrix.at<double>(j,n)) ){
							AGood = true;
						}
						else {
							AGood = false;
						}
						if(outputB[n] > target[n] - sqrt(networkOutputStdDevMatrix.at<double>(k,n)) && outputB[n] < target[n] + sqrt(networkOutputStdDevMatrix.at<double>(k,n)) ){
							BGood = true;
						}
						else {
							BGood = false;
						}
						if(AGood && !BGood){
							disagreementA+=1.0;
						}
						else if(BGood && !AGood){
							disagreementB+=1.0;
						}
						else{
							agreement+=1.0;
						}
					}
				}
			}
			disagreementMatrix.at<double>(j, k) = (disagreementB + disagreementA)/(agreement);
			disagreementMatrix.at<double>(k, j) = (disagreementB + disagreementA)/(agreement);
		}
	}
}

vector<Mat> DiversityMeasurer::getChiSquareMatrix() const {
	return chiSquareMatrix;
}

void DiversityMeasurer::setChiSquareMatrix(vector<Mat> chiSquareMatrix) {
	this->chiSquareMatrix = chiSquareMatrix;
}

vector<Mat> DiversityMeasurer::getCorrelationMatrix() const {
	return correlationMatrix;
}

void DiversityMeasurer::setCorrelationMatrix(vector<Mat> correlationMatrix) {
	this->correlationMatrix = correlationMatrix;
}

vector<Mat> DiversityMeasurer::getCovarianceMatrix() const {
	return covarianceMatrix;
}

void DiversityMeasurer::setCovarianceMatrix(vector<Mat> covarianceMatrix) {
	this->covarianceMatrix = covarianceMatrix;
}

Mat DiversityMeasurer::getDisagreementMatrix() const {
	return disagreementMatrix;
}

RegressionDataset& DiversityMeasurer::getData() const {
	return data;
}

void DiversityMeasurer::setData(RegressionDataset& data) {
	this->data = data;
}

Mat DiversityMeasurer::getNetworkOutputMeanMatrix() const {
	return networkOutputMeanMatrix;
}

void DiversityMeasurer::setNetworkOutputMeanMatrix(Mat networkOutputMeanMatrix) {
	this->networkOutputMeanMatrix = networkOutputMeanMatrix;
}

PBDNN& DiversityMeasurer::getNetworkPopulation() const {
	return networkPopulation;
}

void DiversityMeasurer::setNetworkPopulation(PBDNN& networkPopulation) {
	this->networkPopulation = networkPopulation;
}

Mat DiversityMeasurer::getNetworkOutputStdDevMatrix() const {
	return networkOutputStdDevMatrix;
}


void DiversityMeasurer::setNetworkOutputStdDevMatrix(cv::Mat stdDevMatrix) {
	this->networkOutputStdDevMatrix = stdDevMatrix;
}

void DiversityMeasurer::setDisagreementMatrix(Mat disagreementMatrix) {
	this->disagreementMatrix = disagreementMatrix;
}

DiversityMeasurer::~DiversityMeasurer() {

}
