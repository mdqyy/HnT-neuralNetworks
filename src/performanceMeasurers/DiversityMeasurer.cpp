/*!
 * \file DiversityMeasurer.cpp
 * Body of the DiversityMeasurer class.
 * \author Luc Mioulet
 */

#include "DiversityMeasurer.hpp"

using namespace cv;
using namespace std;

DiversityMeasurer::DiversityMeasurer(PBDNN& _population, RegressionDataset& _data, ErrorMeasurer& _em, realv _percentageOfPopulation) :
		networkPopulation(_population), data(_data), disagreementScalar(0.0), errorMeasurer(_em), percentageOfPopulation(_percentageOfPopulation) {
	initMatrices();
}

vector<uint> DiversityMeasurer::defineIndexOrderSelection(uint _numSequences){
  vector<uint> indexOrder;
  for(uint i=0 ;  i<_numSequences; i++){
    indexOrder.push_back(i);
  }
  int exchangeIndex=0;
  RNG random(getTickCount());
  random.next();
  for(uint i=0 ;  i<_numSequences; i++){
    random.next();
    exchangeIndex=random.uniform(0,_numSequences);
    swap(indexOrder[i],indexOrder[exchangeIndex]);
  }
  return indexOrder;
}

void DiversityMeasurer::measurePerformance() {
	processNetworkOutputMeanAndStdDevMatrix();
	/*processCorrelationMatrix();
	 processCovarianceMatrix();
	 processChiSquareMatrix();*/
	processDisagreementMatrix();
	processDisagreementScalar();
}

void DiversityMeasurer::initMatrices() {
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector output = population[0]->getOutputSignal();
	networkOutputMeanMatrix = Mat::zeros(output.getLength(), population.size(), CV_64FC1);
	networkOutputStdDevMatrix = Mat::zeros(output.getLength(), population.size(), CV_64FC1);
	correlationMatrix = vector<Mat>(networkOutputMeanMatrix.rows, Mat::zeros(population.size(), population.size(), CV_64FC1));
	covarianceMatrix = vector<Mat>(networkOutputMeanMatrix.rows, Mat::zeros(population.size(), population.size(), CV_64FC1));
	chiSquareMatrix = vector<Mat>(networkOutputMeanMatrix.rows, Mat::zeros(population.size(), population.size(), CV_64FC1));
	disagreementMatrix = Mat::zeros(population.size(), population.size(), CV_64FC1);
}

void DiversityMeasurer::processNetworkOutputMeanAndStdDevMatrix() {
	vector<uint> indexOrderSelection = defineIndexOrderSelection(data.getNumSequences());
	uint index = 0;
	FeatureVector networkOutput;
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	Mat test = Mat::zeros(1, 1, CV_64FC1);
	double numSamples = 1.0;
	double oldMean = 0.0;
	for (uint i = 0; i < population.size(); i++) {
		for (uint j = 0; j < data.getNumSequences()*percentageOfPopulation; j++) {
			index = indexOrderSelection[j];
			for (uint k = 0; k < data[index].size(); k++) {
				population[i]->forward(data[index][k]);
				networkOutput = population[i]->getOutputSignal();
				for (uint l = 0; l < networkOutput.getLength(); l++) {
					oldMean = networkOutputMeanMatrix.at<double>(l, i);
					networkOutputMeanMatrix.at<double>(l, i) = networkOutputMeanMatrix.at<double>(l, i)
							+ (networkOutput[l] - networkOutputMeanMatrix.at<double>(l, i)) / numSamples;
					networkOutputStdDevMatrix.at<double>(l, i) = ((numSamples - 1) * networkOutputStdDevMatrix.at<double>(l, i)
							+ (networkOutput[l] - networkOutputMeanMatrix.at<double>(l, i)) * (networkOutput[l] - oldMean)) / numSamples;
					numSamples = numSamples + 1.0;
				}
			}
		}
	}
}

void DiversityMeasurer::processCorrelationMatrix() {
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
						correlationMatrix[n].at<double>(j, k) = correlationMatrix[n].at<double>(j, k)
								+ (outputA[n] - networkOutputMeanMatrix.at<double>(n, j)) * (outputB[n] - networkOutputMeanMatrix.at<double>(n, k));
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

void DiversityMeasurer::processCovarianceMatrix() {
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
						covarianceMatrix[n].at<double>(j, k) = covarianceMatrix[n].at<double>(j, k)
								+ (outputA[n] - networkOutputMeanMatrix.at<double>(n, j)) * (outputB[n] - networkOutputMeanMatrix.at<double>(n, k));
					}
				}
			}
			for (uint n = 0; n < outputA.getLength(); n++) {
				covarianceMatrix[n].at<double>(j, k) = covarianceMatrix[n].at<double>(j, k) / ((double) data.getNumSamples());
				covarianceMatrix[n].at<double>(k, j) = covarianceMatrix[n].at<double>(j, k);
			}
		}
	}
}

void DiversityMeasurer::processChiSquareMatrix() {
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
						chiSquareMatrix[n].at<double>(j, k) = chiSquareMatrix[n].at<double>(j, k)
								+ (outputA[n] - outputB[n]) * (outputA[n] - outputB[n]) / outputB[n];
						chiSquareMatrix[n].at<double>(k, j) = chiSquareMatrix[n].at<double>(j, k)
								+ (outputA[n] - outputB[n]) * (outputA[n] - outputB[n]) / outputA[n];
					}
				}
			}
		}
	}
}

/*void DiversityMeasurer::processDisagreementMatrix() {
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector outputA, outputB, target;
	double agreement, disagreementA, disagreementB;
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
					target = data.getTargetSample(l, m);
					for (uint n = 0; n < outputA.getLength(); n++) {
						if (outputA[n] > target[n] - sqrt(networkOutputStdDevMatrix.at<double>(j, n))
								&& outputA[n] < target[n] + sqrt(networkOutputStdDevMatrix.at<double>(j, n))) {
							AGood = true;
						} else {
							AGood = false;
						}
						if (outputB[n] > target[n] - sqrt(networkOutputStdDevMatrix.at<double>(k, n))
								&& outputB[n] < target[n] + sqrt(networkOutputStdDevMatrix.at<double>(k, n))) {
							BGood = true;
						} else {
							BGood = false;
						}
						if (AGood && !BGood) {
							disagreementA += 1.0;
						} else if (BGood && !AGood) {
							disagreementB += 1.0;
						} else {
							agreement += 1.0;
						}
					}
				}
			}
			disagreementMatrix.at<double>(j, k) = (disagreementB + disagreementA) / (agreement + disagreementB + disagreementA);
			disagreementMatrix.at<double>(k, j) = (disagreementB + disagreementA) / (agreement + disagreementB + disagreementA);
		}
	}
}*/

void DiversityMeasurer::processDisagreementMatrix() {
	vector<NeuralNetworkPtr> population = networkPopulation.getPopulation();
	FeatureVector outputA, outputB, target;
	vector<uint> indexOrderSelection = defineIndexOrderSelection(data.getNumSequences());
	uint index = 0;
	double agreement, disagreementA, disagreementB;
	bool AGood, BGood;
	for (uint l = 0; l < data.getNumSequences()*percentageOfPopulation; l++) {
		index = indexOrderSelection[l];
		for (uint m = 0; m < data[index].size(); m++) {
			target = data.getTargetSample(index, m);
			for (uint j = 0; j < population.size(); j++) {
				population[j]->forward(data[index][m]);
				outputA = population[j]->getOutputSignal();
				for (uint k = j + 1; k < population.size(); k++) {
					agreement = 0.0;
					disagreementA = 0.0;
					disagreementB = 0.0;
					population[k]->forward(data[index][m]);
					outputB = population[k]->getOutputSignal();
					for (uint n = 0; n < outputA.getLength(); n++) {
						if (outputA[n] > target[n] - sqrt(networkOutputStdDevMatrix.at<double>(j, n))
								&& outputA[n] < target[n] + sqrt(networkOutputStdDevMatrix.at<double>(j, n))) {
							AGood = true;
						} else {
							AGood = false;
						}
						if (outputB[n] > target[n] - sqrt(networkOutputStdDevMatrix.at<double>(k, n))
								&& outputB[n] < target[n] + sqrt(networkOutputStdDevMatrix.at<double>(k, n))) {
							BGood = true;
						} else {
							BGood = false;
						}
						if (AGood && !BGood) {
							disagreementA += 1.0;
						} else if (BGood && !AGood) {
							disagreementB += 1.0;
						} else {
							agreement += 1.0;
						}
					}
					disagreementMatrix.at<double>(j, k) = (disagreementB + disagreementA) / (agreement + disagreementB + disagreementA);
					disagreementMatrix.at<double>(k, j) = (disagreementB + disagreementA) / (agreement + disagreementB + disagreementA);
				}
			}
		}
	}
}

void DiversityMeasurer::processDisagreementScalar() {
	disagreementScalar = 0.0;
	realv combinations = 0.0;
	for (int i = 0; i < disagreementMatrix.rows; i++) {
		for (int j = i + 1; j < disagreementMatrix.cols; j++) {
			disagreementScalar += disagreementMatrix.at<double>(i, j);
			combinations += 1.0;
		}
	}
	if (combinations > 0.0) {
		disagreementScalar /= combinations;
	} else {
		disagreementScalar = 0.0;
	}
}

vector<vector<int> > DiversityMeasurer::findBestNetwork() {
	vector<NeuralNetworkPtr> neuralNets = networkPopulation.getPopulation();
	vector<vector<int> > assignedTo = vector<vector<int> >();
	FeatureVector fv;
	realv minError = 10e+9;
	uint bestNetwork = 0;
	for (uint i = 0; i < data.getNumSequences(); i++) {
		vector<int> sequenceAssignement = vector<int>();
		for (uint j = 0; j < data[i].size(); j++) {
			bestNetwork = 0;
			minError = 10e+9;
			fv = data[i][j];
			for (uint k = 0; k < neuralNets.size(); k++) {
				neuralNets[k]->forward(fv);
				errorMeasurer.processErrors(neuralNets[k]->getOutputSignal(), data.getTargetSample(i, j));
				if (errorMeasurer.getError() < minError) {
					minError = errorMeasurer.getError();
					bestNetwork = k;
				}
			}
			sequenceAssignement.push_back(bestNetwork);
		}
		assignedTo.push_back(sequenceAssignement);
	}
	return assignedTo;
}

vector<int> DiversityMeasurer::sampleRepartition() {
	vector<NeuralNetworkPtr> neuralNets = networkPopulation.getPopulation();
	vector<int> clusterSize = vector<int>(neuralNets.size(), 0);
	FeatureVector fv;
	realv minError = 10e+9;
	uint bestNetwork = 0;
	for (uint i = 0; i < data.getNumSequences(); i++) {
		vector<int> sequenceAssignement = vector<int>();
		for (uint j = 0; j < data[i].size(); j++) {
			bestNetwork = 0;
			minError = 10e+9;
			fv = data[i][j];
			for (uint k = 0; k < neuralNets.size(); k++) {
				neuralNets[k]->forward(fv);
				errorMeasurer.processErrors(neuralNets[k]->getOutputSignal(), data.getTargetSample(i, j));
				if (errorMeasurer.getError() < minError) {
					minError = errorMeasurer.getError();
					bestNetwork = k;
				}
			}
			clusterSize[bestNetwork] += 1;
		}
	}
	return clusterSize;
}

vector<realv> DiversityMeasurer::errorsOnBestSample() {
	vector<NeuralNetworkPtr> neuralNets = networkPopulation.getPopulation();
	vector<realv> errors = vector<realv>(neuralNets.size(), 0.0);
	vector<realv> clusterSize = vector<realv>(neuralNets.size(), 0.0);
	FeatureVector fv;
	realv minError = 10e+9;
	uint bestNetwork = 0;
	for (uint i = 0; i < data.getNumSequences(); i++) {
		vector<int> sequenceAssignement = vector<int>();
		for (uint j = 0; j < data[i].size(); j++) {
			bestNetwork = 0;
			minError = 10e+9;
			fv = data[i][j];
			for (uint k = 0; k < neuralNets.size(); k++) {

				neuralNets[k]->forward(fv);
				errorMeasurer.processErrors(neuralNets[k]->getOutputSignal(), data.getTargetSample(i, j));
				if (errorMeasurer.getError() < minError) {
					minError = errorMeasurer.getError();
					bestNetwork = k;
				}
			}
			errors[bestNetwork] += minError;
			clusterSize[bestNetwork] += 1.0;
		}
	}
	for (uint k = 0; k < neuralNets.size(); k++) {
		errors[k] = errors[k] / clusterSize[k];
	}
	return errors;
}

vector<FeatureVector> DiversityMeasurer::getMeanGoodOutput() {
	vector<NeuralNetworkPtr> neuralNets = networkPopulation.getPopulation();
	vector<FeatureVector> meanOutput = vector<FeatureVector>(neuralNets.size(), FeatureVector(data.getTargetSample(0, 0).getLength()));
	FeatureVector fv;
	realv minError = 10e+9;
	uint bestNetwork = 0;
	for (uint i = 0; i < data.getNumSequences(); i++) {
		vector<int> sequenceAssignement = vector<int>();
		for (uint j = 0; j < data[i].size(); j++) {
			bestNetwork = 0;
			minError = 10e+9;
			fv = data[i][j];
			for (uint k = 0; k < neuralNets.size(); k++) {
				neuralNets[k]->forward(fv);
				errorMeasurer.processErrors(neuralNets[k]->getOutputSignal(), data.getTargetSample(i, j));
				if (errorMeasurer.getError() < minError) {
					minError = errorMeasurer.getError();
					bestNetwork = k;
				}
			}
			fv = neuralNets[bestNetwork]->getOutputSignal();
			realv q = 0.0;
			for (int t = 0; t < fv.getLength(); t++) {
				meanOutput[bestNetwork][t] = (q * meanOutput[bestNetwork][t] + fv[t]) / (q + 1.0);
				q += 1.0;
			}
		}
	}
	return meanOutput;
}

vector<vector<FeatureVector> > DiversityMeasurer::buildBestOutput() {
	vector<NeuralNetworkPtr> neuralNets = networkPopulation.getPopulation();
	vector<vector<FeatureVector> > assignedTo = vector<vector<FeatureVector> >();
	FeatureVector fv;
	realv minError = 10e+9;
	uint bestNetwork = 0;
	for (uint i = 0; i < data.getNumSequences(); i++) {
		vector<FeatureVector> sequenceAssignement = vector<FeatureVector>();
		for (uint j = 0; j < data[i].size(); j++) {
			bestNetwork = 0;
			minError = 10e+9;
			fv = data[i][j];
			for (uint k = 0; k < neuralNets.size(); k++) {
				neuralNets[k]->forward(fv);
				errorMeasurer.processErrors(neuralNets[k]->getOutputSignal(), data.getTargetSample(i, j));
				if (errorMeasurer.getError() < minError) {
					minError = errorMeasurer.getError();
					bestNetwork = k;
				}
			}
			sequenceAssignement.push_back(neuralNets[bestNetwork]->getOutputSignal());
		}
		assignedTo.push_back(sequenceAssignement);
	}
	return assignedTo;
}

vector<Mat> DiversityMeasurer::getChiSquareMatrix() const {
	return chiSquareMatrix;
}

void DiversityMeasurer::setChiSquareMatrix(vector<Mat> _chiSquareMatrix) {
	this->chiSquareMatrix = _chiSquareMatrix;
}

vector<Mat> DiversityMeasurer::getCorrelationMatrix() const {
	return correlationMatrix;
}

void DiversityMeasurer::setCorrelationMatrix(vector<Mat> _correlationMatrix) {
	this->correlationMatrix = _correlationMatrix;
}

vector<Mat> DiversityMeasurer::getCovarianceMatrix() const {
	return covarianceMatrix;
}

void DiversityMeasurer::setCovarianceMatrix(vector<Mat> _covarianceMatrix) {
	this->covarianceMatrix = _covarianceMatrix;
}

Mat DiversityMeasurer::getDisagreementMatrix() const {
	return disagreementMatrix;
}

RegressionDataset& DiversityMeasurer::getData() const {
	return data;
}

void DiversityMeasurer::setData(RegressionDataset& _data) {
	this->data = _data;
}

Mat DiversityMeasurer::getNetworkOutputMeanMatrix() const {
	return networkOutputMeanMatrix;
}

void DiversityMeasurer::setNetworkOutputMeanMatrix(Mat _networkOutputMeanMatrix) {
	this->networkOutputMeanMatrix = _networkOutputMeanMatrix;
}

PBDNN& DiversityMeasurer::getNetworkPopulation() const {
	return networkPopulation;
}

void DiversityMeasurer::setNetworkPopulation(PBDNN& _networkPopulation) {
	this->networkPopulation = _networkPopulation;
}

Mat DiversityMeasurer::getNetworkOutputStdDevMatrix() const {
	return networkOutputStdDevMatrix;
}

void DiversityMeasurer::setNetworkOutputStdDevMatrix(cv::Mat _stdDevMatrix) {
	this->networkOutputStdDevMatrix = _stdDevMatrix;
}

void DiversityMeasurer::setDisagreementMatrix(Mat _disagreementMatrix) {
	this->disagreementMatrix = _disagreementMatrix;
}

realv DiversityMeasurer::getDisagreementScalar() const {
	return disagreementScalar;
}

void DiversityMeasurer::setDisagreementScalar(realv _disagreementScalar) {
	this->disagreementScalar = _disagreementScalar;
}

ErrorMeasurer& DiversityMeasurer::getErrorMeasurer() const {
	return errorMeasurer;
}

DiversityMeasurer::~DiversityMeasurer() {

}
