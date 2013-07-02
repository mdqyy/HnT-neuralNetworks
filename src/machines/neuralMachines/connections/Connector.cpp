#include "Connector.hpp"


using namespace std;
using namespace cv;

Connector::Connector() : layers(vector<LayerPtr>()),length(0){
	
}

Connector::Connector(vector<LayerPtr> _outputLayers) : layers(_outputLayers){
  length = 0;
  for (uint i = 0; i < layers.size(); i++){
    length += layers[i]->getNumUnits();
  }
}


uint Connector::getLength(){
  return length;
}

FeatureVector Connector::concatenateOutputs(){
  FeatureVector result(length);
  uint index=0;
  for (uint i = 0; i < layers.size(); i++){
    FeatureVector fv= layers[i]->getOutputSignal();
    for(uint j=0;j<fv.getLength()-1;j++){ /* -1 to avoid using the bias , yeah I implemented poorly this bias ...*/
      result[index] = fv[j];
      index++;
    }
  }
  return result;
}

