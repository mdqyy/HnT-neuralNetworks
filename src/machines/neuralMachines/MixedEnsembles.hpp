#ifndef __MIXEDENSEMBLES_HPP__
#define __MIXEDENSEMBLES_HPP__
/*!
 * \file MixedEnsembles.hpp
 * Header of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include "NeuralNetwork.hpp"
#include "connections/Connection.hpp"
#include "connections/Connector.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/LayerSigmoid.hpp"

/*!
 * \class MixedEnsembles
 * Description
 */
class MixedEnsembles : private NeuralMachine{
 private :

 protected:
  /*! population of networks */
  std::vector<NeuralNetworkPtr> networks;
  
  /*! Ife corresponding to each network */
  std::vector<ImageFrameExtractor> ifes;

  std::vector<uint> linkedToIFE;

  /* Connector media */
  Connector connector;
  
  /* Final output network */
  NeuralNetwork outputNetwork;

 public:

  /*!
   * Default constructor.
   */
  MixedEnsembles(std::vector<NeuralNetworkPtr> _networks,std::vector<ImageFrameExtractor> _ifes, std::vector<uint> _linkedToIFE,Connector _connector, NeuralNetwork _outputNetwork);

  /*!
   * Forward a sequence.
   * \param _sequence Sequence to pass forward.
   */
  void forwardMatrix(cv::Mat _matrix);

  /*!
   * Forward a sequence.
   * \param _sequence Sequence to pass forward.
   */
  void forwardSequence(std::vector<FeatureVector> _sequence);
  
  /*!
   * Forward a sample of a sequence.
   * \param _sample Feature vector to pass forward.
   */
  void forward(FeatureVector _sample);

  /*!
   * Destructor.
   */
  ~MixedEnsembles();

};


#endif
