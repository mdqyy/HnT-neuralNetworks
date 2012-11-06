#ifndef __POPULATIONBPBATCH_HPP__
#define __POPULATIONBPBATCH_HPP__
/*!
 * \file PopulationBPBatch.hpp
 * Header of the PopulationBPBatch class.
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
#include "../../../dataset/supervised/RegressionDataset.hpp"
#include "LearningParams.hpp"
#include <boost/thread/thread.hpp>

/*!
 * \class PopulationBPBatch
 * Description
 */
class PopulationBPBatch : public SupervisedTrainer{
 private :

 protected:
  /*! Network population */
  PBDNN& population;
  /*! Learning parameters */
  LearningParams params;
  /*! Regression dataset*/
  RegressionDataset regData;
 public:

  /*!
   * Parameter constructor.
   * \param _population The neural population.
   * \param _data The training data.
   * \param _params The training parameters.
   * \param _featureMask Feature mask.
   * \param _indexMask Index mask.
   */
  PopulationBPBatch(PBDNN& _population, RegressionDataset& _data, LearningParams& _params, Mask& _featureMask, Mask& _indexMask);

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
  ~PopulationBPBatch();

};


#endif
