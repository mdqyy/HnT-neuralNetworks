#ifndef __LEARNINGPARAMS_HPP__
#define __LEARNINGPARAMS_HPP__
/*!
 * \file LearningParams.hpp
 * Header of the LearningParams class.
 * \author Luc Mioulet
 */

#include "../../../General.hpp"
#include <opencv/cv.h>
#include <stdio.h>
#include <fstream>
#include <ostream>
#include <istream>
#include <sstream>

#define BP_CLASSIFICATION 0
#define BP_REGRESSION 1
#define BP_AUTOENCODER 2

/*!
 * \class LearningParams
 * Learning parameters for neural networks.
 */
class LearningParams {
private:

protected:
	/*! Do stochastic learning (randomize sequence selection).*/
	bool stochastic;

	/*! Learning rate for weight change. */
	realv learningRate;

	/*! Learning rate decrease between each step. */
	realv learningRateDecrease;

	/*! Maximum number of iterations */
	uint maxIterations;

	/*! actualIteration */
	uint actualIteration;

	/*! Minimum error change between two steps*/
	realv minError;

	/*! Minimum error change between two iterations */
	realv minChangeError;

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

	/*! Validate during process */
	bool validatedDuringProcess;
	
	/*! Validation steps every n iteration */
	int validateEveryNIteration;

	/*! Problem type (BP_CLASSIFICATION, BP_REGRESSION, BP_AUTOENCODER) */
	int task;

	/*! Number of consecutive times a population can have an element learn no training data*/
	uint dodges;

	/* Proximity */
	realv proximity;

public:

	/*!
	 * Default constructor.
	 */
	LearningParams(realv _learningRate = 0.001, realv _learningRateDecrease = 0.95, uint _maxIterations = 10, uint _actualIteration = 0, uint _maxTrained = 5,
			realv _maxTrainedPercentage = 0.25, realv _errorToFirst = 0.5, realv errorToFirstIncrease = 1.1, bool _savedDuringProcess = false,
		       std::string _saveLocation = ".", bool _validatedDuringProcess = true, int _validateEveryNIteration=5, int _dodges = 2, realv proximity = 0.30);

	/*!
	 * Do stochastic.
	 * \return Do stochastic.
	 */
	bool getDoStochastic();

	/*!
	 * Get the minimum error to be achieved.
	 * \return Minimum error to be achieved.
	 */
	realv getMinError();

	/*!
	 * Get minimum error change for which the backpropagation shoud be stopped.
	 * \return Minimum error change.
	 */
	realv getMinChangeError();

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
	 * Get number of dodges.
	 * \return Number of dodges.
	 */
	uint getDodges() const;

	/*!
	 * Get maximum percentage of trained samples over one iteration.
	 */
	realv getMaxTrainedPercentage() const;

	/*!
	 * Get save location.
	 * \return Save location.
	 */
	std::string getSaveLocation() const;

	/*!
	 * Get proximity .
	 * \return Proximity.
	 */
	realv getProximity() const;

	/*!
	 * Get validated during process.
	 * \return Is validated during learning.
	 */
	bool isValidatedDuringProcess() const;

	/*!
	 * Get saved during process.
	 * \return Is saved during learning.
	 */
	bool isSavedDuringProcess() const;

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
	 * Set maximum percentage of trained samples over one iteration.
	 */
	void setMaxTrainedPercentage(realv _maxTrainedPercentage);

	/*!
	 * Set saved during process.
	 * \param Is saved during learning.
	 */
	void setSavedDuringProcess(bool _savedDuringProcess);


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
	 * Get the task.
	 * \return Get the processing task.
	 */
	int getTask() const;

	/*!
	 * Set actual iteration.
	 * \param _actualIteration Actual iteration.
	 */
	void setActualIteration(uint _actualIteration);

	/*!
	 * Set validated during process.
	 * \param Is validated during learning.
	 */
	void setValidatedDuringProcess(bool _validatedDuringProcess);

	/*!
	 * Set the minimum change error.
	 * \param _minChangeError Minimum change in error before stopping learning.
	 */
	void setMinChangeError(realv _minChangeError);

	/*!
	 * Set the minimum error.
	 * \param _minError Minimum error to achieve before stopping learning.
	 */
	void setMinError(realv _minError);

	/*!
	 * Get if the process is set as stochastic (random selection of a subpart of the dataset).
	 * \return Is stochastic.
	 */
	bool isStochastic() const;

	/*!
	 * Set the learning to be stochastic. Also change the maximum percentage of trained samples.
	 * \param _stochastic True or false.
	 */
	void setStochastic(bool _stochastic);

	/*!
	 * Set the task.
	 * \param _task The learning task.
	 */
	void setTask(int _task);

	/*! 
	 * Set the number of dodges.
	 * \param _dodges The number of doges.
	 */
	void setDodges(uint _dodges);

	/*!
	 * Set the proximity.
	 * \param _proximity The proximity as a precentage.
	 */
	void setProximity(realv _proximity);

	/*!
	 * Destructor.
	 */
	~LearningParams();

	/*!
	 * Output file stream.
	 * \param _ofs Output file stream.
	 * \param _p Population Backprop networks.
	 * \return Output file stream.
	 */
	friend std::ofstream& operator<<(std::ofstream& _ofs, const LearningParams& _p);

	/*!
	 * Input file stream.
	 * \param _ifs Input file stream.
	 * \param _p Population Backprop networks.
	 * \return Input file stream.
	 */
	friend std::ifstream& operator>>(std::ifstream& _ifs, LearningParams& _p);
	int getValidateEveryNIteration() const;
	void setValidateEveryNIteration(int validateEveryNIteration);


}
;

#endif
