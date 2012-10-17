#ifndef __POPULATIONBPPARAMS_HPP__
#define __POPULATIONBPPARAMS_HPP__
/*!
 * \file PopulationBPParams.hpp
 * Header of the PopulationBPParams class.
 * \author Luc Mioulet
 */

#include "../../../General.hpp"
#include <opencv/cv.h>
#include <stdio.h>
//#include <iostream>
#include <fstream>
#include <ostream>
#include <istream>
#include <sstream>

#define BP_CLASSIFICATION 0
#define BP_REGRESSION 1
#define BP_AUTOENCODER 2

/*!
 * \class PopulationBPParams
 * Description
 */
class PopulationBPParams {
private:

protected:
	/*! Learning rate for weight change. */
	realv learningRate;

	/*! Learning rate decrease between each step. */
	realv learningRateDecrease;

	/*! Maximum number of iterations */
	uint maxIterations;

	/*! actualIteration */
	uint actualIteration;

	/*! Maximum number of networks to train per iteration*/
	uint maxTrained;

	/*! Maximum percentage of trained samples (range from 0 to 1) */
	realv maxTrainedPercentage;

	/*! Error ratio between best classifier and worst selected for training.*/
	realv errorToFirst;

	/*! Error ratio increase */
	realv errorToFirstIncrease;

	/*! Save during process */
	bool savedDuringProcess;

	/*! Save location */
	std::string saveLocation;

public:

	/*!
	 * Default constructor.
	 */
	PopulationBPParams(realv _learningRate = 0.001, realv _learningRateDecrease = 0.95, uint _maxIterations = 10, uint _actualIteration = 0, uint _maxTrained = 5,
			realv _maxTrainedPercentage = 0.25, realv _errorToFirst = 0.5, realv errorToFirstIncrease = 1.1, bool _savedDuringProcess = false, std::string _saveLocation =".");

	/*!
	 * Get learning rate.
	 * \return Weight change rate per iteration.
	 */
	realv getLearningRate() const;

	/*!
	 * Get Learning rate decrease.
	 * \return Weight decrease.
	 */
	realv getLearningRateDecrease() const;

	/*!
	 * Get maximum number of iterations.
	 * \return Maximum number of iterations.
	 */
	uint getMaxIterations() const;

	/*!
	 * Get the maxmimum number of networks to train at each iteration.
	 * \return Maximum of netwoks trained.
	 */
	uint getMaxTrained() const;

	/*!
	 * Get error to best network selected.
	 * \return Error to best network.
	 */
	realv getErrorToFirst() const;

	/*!
	 * Get ratio increase to best network selected.
	 * \return Increase ratio to best network.
	 */
	realv getErrorToFirstIncrease() const;

	/*!
	 * Set learning rate.
	 * \param _learningRate New learning rate.
	 */
	void setLearningRate(realv _learningRate);

	/*!
	 * Set learning rate decrease.
	 * \param _learningRateDecrease Learing rate decrease.
	 */
	void setLearningRateDecrease(realv _learningRateDecrease);

	/*!
	 * Set maximum iterations.
	 * \param _maxIterations Maximum number of iterations.
	 */
	void setMaxIterations(uint _maxIterations);

	/*!
	 * Set maximum of networks to train.
	 * \param _maxTrained Maximum of networks to train.
	 */
	void setMaxTrained(uint _maxTrained);

	/*!
	 * Set the error to first ratio.
	 * \param _errorToFirst Error to first.
	 */
	void setErrorToFirst(realv _errorToFirst);

	/*!
	 * Set the increase for the ratio to first.
	 * \param _errorToFirstIncrease Increase.
	 */
	void setErrorToFirstIncrease(realv _errorToFirstIncrease);

	/*!
	 * Get maximum percentage of trained samples over one iteration.
	 */
	realv getMaxTrainedPercentage() const;

	/*!
	 * Set maximum percentage of trained samples over one iteration.
	 */
	void setMaxTrainedPercentage(realv _maxTrainedPercentage);

	/*!
	 * Get save during process.
	 * \return Is saved during learning.
	 */
	bool isSavedDuringProcess() const;

	/*!
	 * Set save during process.
	 * \param Is saved during learning.
	 */
	void setSavedDuringProcess(bool _savedDuringProcess);

	/*!
	 * Get save location.
	 * \return Save location.
	 */
	std::string getSaveLocation() const;

	/*!
	 * Set save location
	 * \param _saveLocation Save location.
	 */
	void setSaveLocation(std::string _saveLocation);

	/*!
	 * Get actual iteration.
	 * \return Actual iteration.
	 */
	uint getActualIteration() const;

	/*!
	 * Set actual iteration.
	 * \param _actualIteration Actual iteration.
	 */
	void setActualIteration(uint _actualIteration);


	/*!
	 * Destructor.
	 */
	~PopulationBPParams();

	/*!
	 * Output file stream.
	 * \param _ofs Output file stream.
	 * \param _p Population Backprop networks.
	 * \return Output file stream.
	 */
	friend std::ofstream& operator<<(std::ofstream& _ofs, const PopulationBPParams& _p);

	/*!
	 * Input file stream.
	 * \param _ifs Input file stream.
	 * \param _p Population Backprop networks.
	 * \return Input file stream.
	 */
	friend std::ifstream& operator>>(std::ifstream& _ifs, PopulationBPParams& _p);
};

#endif
