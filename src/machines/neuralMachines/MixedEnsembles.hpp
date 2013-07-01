#ifndef __MIXEDENSEMBLES_HPP__
#define __MIXEDENSEMBLES_HPP__
/*!
 * \file MixedEnsembles.hpp
 * Header of the MixedEnsembles class.
 * \author Luc Mioulet
 */

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
class MixedEnsembles {
 private :

 protected:
  std::vector<NeuralNetworkPtr> networks;
  std::vector<ImageFrameExtractor> ifes;
  Connector connector;

 public:

  /*!
   * Default constructor.
   */
  MixedEnsembles();

  /* Todo : Methods*/

  /*!
   * Destructor.
   */
  ~MixedEnsembles();

};


#endif
