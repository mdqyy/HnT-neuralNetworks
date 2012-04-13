#ifndef __POPULATIONBP_HPP__
#define __POPULATIONBP_HPP__
/*!
 * \file PopulationBP.hpp
 * Header of the PopulationBP class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include <vector>
#include <list>
#include "../SupervisedTrainer.hpp"
#include "../../../machines/neuralMachines/PBDNN.hpp"
#include "../../../machines/neuralMachines/NeuralNetwork.hpp"
#include "PopulationBPParams.hpp"
#include <boost/thread/thread.hpp>

/*!
 * \class PopulationBP
 * Description
 */
class PopulationBP : public SupervisedTrainer{
 private :

 protected:
  /*! Network population */
  PBDNN& population;
  /*! Learning parameters */
  PopulationBPParams params;

 public:

  /*!
   * Parameter constructor.
   * \param _population The neural population.
   * \param _data The training data.
   * \param _params The training parameters.
   * \param _featureMask Feature mask.
   * \param _indexMask Index mask.
   */
  PopulationBP(PBDNN& _population, SupervisedDataset& _data, PopulationBPParams& _params, Mask& _featureMask, Mask& _indexMask);

  /*!
   * Train the neural networks.
   */
  virtual void train();

  /*! 
   * Pretrain a population
   */
  void preTrain();

  /*!
   * Train the neural networks on one iteration.
   */
  void trainOneIteration();

  /*!
   * Destructor.
   */
  ~PopulationBP();

};


#endif
