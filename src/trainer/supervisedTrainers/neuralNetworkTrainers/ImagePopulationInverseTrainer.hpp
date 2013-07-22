#ifndef __IMAGEPOPULATIONINVERSETRAINER_HPP__
#define __IMAGEPOPULATIONINVERSETRAINER_HPP__
/*!
 * \file ImagePopulationInverseTrainer.hpp
 * Header of the Population InverseTrainer class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <vector>
#include <list>
#include "../../../dataset/ImageDataset.hpp"
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
 * \class ImagePopulationInverseTrainer
 * Description
 */
class ImagePopulationInverseTrainer {
  private:

  protected:
    /*! Network population */
    PBDNN& population;
    /*! Regression dataset*/
    ImageDataset trainingDataset;
    /*! Diversity measurer */
    ImageDataset validationDataset;
    /*! Learning parameters */
    LearningParams params;
    /*Log */
    std::ostream& log;
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
    ImagePopulationInverseTrainer(PBDNN& _population, ImageDataset& _trainDataset, ImageDataset& _validationDataset, LearningParams& _params, std::ostream& _log);

    /*!
     * Noise an input;
     * \params _fv input feature vector.
     */
    FeatureVector noiseTarget(FeatureVector _fv);

    /*!
     * Used to define the index order call of the different sequences during learning.
     * \param _numSequences Number of sequences in the data.
     * \return Vector of unsigned integers.
     */
    std::vector<uint> defineIndexOrderSelection(uint _numSequences);

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
    std::vector<std::vector<uint> > determineLearningAffectations(std::vector<
        std::vector<realv> >& _errors, std::vector<uint>& _index, uint _numberOfElementsToProcess, realv _maxError);

    /*!
     * Destructor.
     */
    ~ImagePopulationInverseTrainer();
};

#endif
