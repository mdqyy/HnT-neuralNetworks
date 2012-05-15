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

  ErrorVector calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta);
  ErrorVector calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives);
  void updateConnection(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate);
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
   * Backward errors from one target.
   * \param _neuralNet Network being backpropagated.
   * \param _target Target feature vector.
   * \param _learningRate Learning rate for weight changes.
   */
  void backward(NeuralNetworkPtr _neuralNet,FeatureVector _target, realv _learningRate);


  /*!
   * Destructor.
   */
  ~PopulationBP();

};


#endif
