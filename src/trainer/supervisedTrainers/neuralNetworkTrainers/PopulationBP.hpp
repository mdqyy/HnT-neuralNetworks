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

/*!
 * \class PopulationBP
 * Description
 */
class PopulationBP : public SupervisedTrainer{
 private :

 protected:
  PBDNN& population;

 public:

  /*!
   * Default constructor.
   */
  PopulationBP(PBDNN& _population, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask);

  /*!
   * Train the neural networks.
   */
  virtual void train();

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
