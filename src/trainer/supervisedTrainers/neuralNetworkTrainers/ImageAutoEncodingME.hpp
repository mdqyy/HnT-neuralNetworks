#ifndef __IMAGEAUTOENCODINGME_HPP__
#define __IMAGEAUTOENCODINGME_HPP__
/*!
 * \file ImageAutoEncodingME.hpp
 * Header of the ImageAutoEncodingME class.
 * \author Luc Mioulet
 */

#include "../../../machines/Machine.hpp"
#include "../../../dataset/ImageDataset.hpp"
#include "../../../dataset/ErrorVector.hpp"
#include "../../../dataset/FeatureVector.hpp"
#include "../../../dataset/ValueVector.hpp"
#include "../../../machines/neuralMachines/layers/Layer.hpp"
#include "../../../machines/neuralMachines/MixedEnsembles.hpp"
#include "../../../machines/neuralMachines/connections/Connection.hpp"
#include "../../errorMeasurers/AEMeasurer.hpp"
#include <vector>
#include <opencv/cv.h>
#include "LearningParams.hpp"

/*!
 * \class ImageAutoEncodingME
 * Description
 */
class ImageAutoEncodingME {
 private :

 protected:
  /*! Machine to be trained */
  MixedEnsembles& machine;

  /* Image dataset */
  ImageDataset& dataset;

  /* Image test dataset */
  ImageDataset& testDataset;

  /* Learning parameters */
  LearningParams params;
  
  /*! Logging stream*/
  std::ostream& log;

  /*! Backward error in the network
   * \param _target The output target.
   */
  void backward(FeatureVector _target);

  /*! Update connection weights.
   * \param _connection The input connection.
   * \param _deltas The error vector.
   */
  void updateConnection(ConnectionPtr _connection, ErrorVector _deltas);

  /*! Calculate the deltas of the output layer.
   * \param _layer The last layer.
   * \param _target The output target.
   * \param _derivatives The derivative values.
   * \return The output error vector.
   */
  ErrorVector calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives);

  /*! Calculate the deltas of the output layer.
   * \param _layer The last layer.
   * \param _target The output target.
   * \param _derivatives The derivative values.
   * \param _previousLayerDelta Previous layer error vector.
   * \return The output error vector.
   */
  ErrorVector calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta);
 public:

  /*!
   * Default constructor.
   */
  ImageAutoEncodingME(MixedEnsembles& _machine, ImageDataset& _dataset, ImageDataset& _testDataset, LearningParams _params, std::ostream& _log);

  /*!
   * Train on a number of iterations.
   */
  void train();
  
  /*!
   * Single training step.
   */
  void trainOneIteration();

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

  /*
   * Validate an iteration.
   */
  void validateIteration();

  /*!
   * Destructor.
   */
  ~ImageAutoEncodingME();

};


#endif
