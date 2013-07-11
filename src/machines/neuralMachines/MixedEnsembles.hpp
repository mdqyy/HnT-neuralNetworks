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
  NeuralNetworkPtr outputNetwork;

 public:
  /*
   * Default constructor.
   */
  MixedEnsembles();

  /*!
   * Parameter constructor.
   */
  MixedEnsembles(std::vector<NeuralNetworkPtr> _networks,std::vector<ImageFrameExtractor> _ifes, std::vector<uint> _linkedToIFE, NeuralNetworkPtr _outputNetwork);

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
   * Get the output of the connector layer.
   * The connector output is not passed forward to the output network.
   * \param _matrix Image to use.
   * \param _i Row pixel.
   */
  FeatureVector getConnectorOutput(cv::Mat _matrix,uint _i);

  /*!
   * Get the output network
   * \return the output network.
   */
  NeuralNetworkPtr getOutputNetwork() const ;

  /*!
   * Set the output network
   * \param _outputNet The output network.
   */
  void setOutputNetwork(NeuralNetworkPtr _outputNet);

  /* Get input networks.
   * \return The input networks.
   */
  std::vector<NeuralNetworkPtr> getNetworks() const;

  /*!
   * Set the input networks
   * \param _networks The input networks.
   */
  void setNetworks(std::vector<NeuralNetworkPtr> _networks);

  /* Get the image frame extractors.
   * \return The image frame extractors.
   */
  std::vector<ImageFrameExtractor> getIFEs() const;

  /*!
   * Set the Image frame extractors used.
   * \param _ifes The new Image frame extractor.
   */
  void setIFEe(std::vector<ImageFrameExtractor> _ifes);

  /* Get the linked to image frame extractor.
   * \return The links to the image frame extractors.
   */
  std::vector<uint> getLinkedToIFE() const;

  /*!
   * Set the links to IFE
   * \param _linkedToIFE The links to IFE.
   */
  void setLinkedToIFE(std::vector<uint> _linkedToIFE);

  /*!
   * Destructor.
   */
  ~MixedEnsembles();

  friend std::ofstream& operator<<(std::ofstream& _ofs, const MixedEnsembles& _ensemble);

  friend std::ifstream& operator>>(std::ifstream& _ifs, MixedEnsembles& _pop);

};


#endif
