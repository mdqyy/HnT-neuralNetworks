#ifndef __CONNECTOR_HPP__
#define __CONNECTOR_HPP__

#include<vector>
#include "../layers/Layer.hpp"
#include "../../../dataset/FeatureVector.hpp"
#include <opencv/cv.h>

class Connector{
 public:
		
  /*!
   * Create a connector class. Will not do anything.
   */
  Connector();
  		
  /*!
   * Create a connector class.
   * \param _outputLayers The vector of target output layers.
   */
  Connector(std::vector<LayerPtr> _outputLayers);

  /*!
   * Get length.
   * \return Connector length
   */
  uint getLength();
		
  /*!
   * Concatenate all vectors.
   * \return A concatenated featureVector.
   */
  FeatureVector concatenateOutputs();
 private:
  std::vector<LayerPtr> layers;

  uint length;
};

#endif /* CONNECTOR_HPP */ 
