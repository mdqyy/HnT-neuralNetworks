#ifndef __MIXEDENSEMBLES_HPP__
#define __MIXEDENSEMBLES_HPP__
/*!
 * \file MixedEnsembles.hpp
 * Header of the MixedEnsembles class.
 * \author Luc Mioulet
 */

#include <opencv/cv.h>
#include "NeuralMachine.hpp"
#include "NeuralNetwork.hpp"
#include "connections/Connection.hpp"
#include "connections/Connector.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/LayerSigmoid.hpp"
#include "../../utilities/ImageFrameExtractor.hpp"
#include <vector>
#include <iostream>
#include <boost/thread/thread.hpp>

/*!
 * \class MixedEnsembles
 * Description
 */
class MixedEnsembles{
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
   * Forward on frames extracted around a pixel.
   * \param _matrix Image to use.
   * \param _i Row pixel.
   */
  void forwardOnPixel(cv::Mat _matrix, uint _i);

  /*!
   * Destructor.
   */
  ~MixedEnsembles();

};


#endif
