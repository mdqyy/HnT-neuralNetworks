#ifndef __POPULATIONCLUSTERBP_HPP__
#define __POPULATIONCLUSTERBP_HPP__
/*!
 * \file PopulationClusterBP.hpp
 * Header of the PopulationClusterBP class.
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
#include "PopulationBPParams.hpp"
#include <boost/thread/thread.hpp>
#include <iostream>
#include <sstream>
#include <ostream>
#include <stdio.h>

/*!
 * \class PopulationClusterBP
 * Description
 */
class PopulationClusterBP: public SupervisedTrainer {
private:

protected:
	/*! Network population */
	PBDNN& population;
	/*! Learning parameters */
	PopulationBPParams params;
	/*! Regression dataset*/
	RegressionDataset regData;
	/*! Diversity measurer */
	RegressionDataset validationDataset;

public:

	/*!
	 * Parameter constructor.
	 * \param _population The neural population.
	 * \param _data The training data.
	 * \param _params The training parameters.
	 * \param _featureMask Feature mask.
	 * \param _indexMask Index mask.
	 */
	PopulationClusterBP(PBDNN& _population, RegressionDataset& _data, PopulationBPParams& _params, RegressionDataset& _valid, Mask& _featureMask, Mask& _indexMask);

	/*!
	 * Train the neural networks.
	 */
	void train();

	/*!
	 * Train the neural networks on one iteration.
	 */
	void trainOneIteration();

	/*!
	 * Destructor.
	 */
	~PopulationClusterBP();
};

#endif
