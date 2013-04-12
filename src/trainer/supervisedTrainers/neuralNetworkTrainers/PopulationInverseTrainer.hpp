#ifndef __POPULATIONINVERSETRAINER_HPP__
#define __POPULATIONINVERSETRAINER_HPP__
/*!
 * \file PopulationInverseTrainer.hpp
 * Header of the Population InverseTrainer class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <vector>
#include <list>
#include "../SupervisedTrainer.hpp"
#include "../../../machines/neuralMachines/PBDNN.hpp"
#include "../../../machines/neuralMachines/NeuralNetwork.hpp"
#include "../../../performanceMeasurers/DiversityMeasurer.hpp"
#include "../../../performanceMeasurers/RegressionMeasurer.hpp"
#include "LearningParams.hpp"
#include <boost/thread/thread.hpp>
#include <iostream>
#include <sstream>
#include <ostream>
#include <stdio.h>

/*!
 * \class PopulationInverseTrainer
 * Description
 */
class PopulationInverseTrainer: public SupervisedTrainer {
private:

protected:
	/*! Network population */
	PBDNN& population;
	/*! Learning parameters */
	LearningParams params;
	/*! Regression dataset*/
	RegressionDataset regData;
	/*! Diversity measurer */
	RegressionDataset validationDataset;
	/*! Survivability, number of aegis */
	std::vector<uint> endurance;


public:

	/*!
	 * Parameter constructor.
	 * \param _population The neural population.
	 * \param _data The training data.
	 * \param _params The training parameters.
	 * \param _valid The validation dataset.
	 * \param _featureMask Feature mask.
	 * \param _indexMask Index mask.
	 *
	 */
	PopulationInverseTrainer(PBDNN& _population, RegressionDataset& _data, LearningParams& _params, RegressionDataset& _valid, Mask& _featureMask, Mask& _indexMask, std::ostream& _log);

	/*!
	 * Train the neural networks.
	 */
	void train();

	/*!
	 * Train the neural networks on one iteration.
	 */
	void trainOneIteration();

	/*!
	 * Validate against a reference dataset.
	 */
	void validateIteration();

	/*!
	 * Determine the learning affectations of all chosen indexes.
	 * \param _errors Errors from the different learning examples
	 * \param _index Index of the elements.
	 * \param _numberOfElementsToProcess Number of elements processed.
	 */
	std::vector<std::vector<uint > > determineLearningAffectations(std::vector<std::vector<realv> >& _errors, std::vector<uint>& _index , uint _numberOfElementsToProcess, realv _maxError);

	/*!
	 * Destructor.
	 */
	~PopulationInverseTrainer();
};

#endif
