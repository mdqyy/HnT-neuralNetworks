#ifndef __NEURALNETWORKTRAINER_HPP__
#define __NEURALNETWORKTRAINER_HPP__
/*!
 * \file NeuralNetworkTrainer.hpp
 * Header of the NeuralNetworkTrainer class.
 * \author Luc Mioulet
 */

#include "../SupervisedTrainer.hpp"
#include "../../../machines/neuralMachines/NeuralNetwork.hpp"
#include <vector>
#include <opencv/cv.h>

/*!
 * \class NeuralNetworkTrainer
 * Description
 */
class NeuralNetworkTrainer : public SupervisedTrainer {
 private :

 protected:
  NeuralNetwork& neuralNet;

 public:

  /*!
   * Parameter constructor.
   * \param _neuralNet NeuralNetwork to train.
   * \param _data Supervised dataset to use for learning.
   
   */
  NeuralNetworkTrainer(NeuralNetwork& _neuralNet, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask, std::ostream& _log);

  /*!
   * Train the machine.
   */
  virtual void train() =0;

  /*!
   * Destructor.
   */
  ~NeuralNetworkTrainer();

};


#endif
