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
  /*! Defined to true when randomization of sequences is required. */
  bool doStochastic;

 public:

  /*!
   * Parameter constructor.
   * \param _neuralNet NeuralNetwork to train.
   * \param _data Supervised dataset to use for learning.
   
   */
  NeuralNetworkTrainer(NeuralNetwork& _neuralNet, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask, bool _doStochastic=true );

  /*!
   * Used to define the index order call of the different sequences during learning.
   * \param _numSequences Number of sequences in the data.
   * \return Vector of unsigned integers.
   */
  std::vector<uint> defineIndexOrderSelection(uint _numSequences);

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
