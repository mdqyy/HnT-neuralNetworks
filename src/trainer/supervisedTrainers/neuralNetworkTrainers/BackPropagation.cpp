/*!
 * \file BackPropagation.cpp
 * Body of the BackPropagation class.
 * \author Luc Mioulet
 */

#include "BackPropagation.hpp"
#include <iostream>

using namespace cv;
using namespace std;

BackPropagation::BackPropagation(NeuralNetwork& _neuralNet, SupervisedDataset& _data, BackPropParams& _bpparams, Mask& _featureMask, Mask& _indexMask) : NeuralNetworkTrainer(_neuralNet, _data, _featureMask, _indexMask, _bpparams.getDoStochastic()), bpp(_bpparams){ 
  
}

void BackPropagation::train(){
  uint i=0;
  realv change=bpp.getMinChangeError()+1.0;
  do{
    i++;
    trainOneIteration();
    if(i>2){
      change=abs(errorPerIteration[i-2]-errorPerIteration[i-1]);
    }
    bpp.setLearningRate(bpp.getLearningRate()*bpp.getLearningRateDecrease());
    cout << "Iteration : "<< i << " ; Error : " << errorPerIteration[i-1]<< endl;
    /*    if(i%bpp.getValidationSteps()==0){ does not work (Floating point exception)
      //      measurePerformance(getValidationDataset());
      }*/
  }while(i<bpp.getMaxIterations() && change>bpp.getMinChangeError() && errorPerIteration[i-1]>bpp.getMinError());
}

void BackPropagation::trainOneIteration(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  uint index=0;
  realv error= 0;
  MSEMeasurer mse;
  FeatureVector dataOutput;
  for(uint i=0; i<data.getNumSequences();i++){
    index=indexOrderSelection[i];
    if(neuralNet.isForward()){
      for(uint j=0;j<data[index].size() ; j++){
	neuralNet.forward(data[index][j]);
	FeatureVector target=trainData.getTargetSample(index,j);
	error += mse.totalError(neuralNet.getOutputSignal(),target);
	neuralNet.backward(target, bpp.getLearningRate());
      }
    }
    else{
      for(uint j=data.getNumSequences();j>=0 ; j--){
	neuralNet.forward(data[index][j]);
	FeatureVector target=trainData.getTargetSample(index,j);
	error += mse.totalError(neuralNet.getOutputSignal(),target);
	neuralNet.backward(target, bpp.getLearningRate());
      }
    }
  }
  cout << error/data.getNumSequences() << endl;
  errorPerIteration.push_back(error/data.getNumSequences());
}



BackPropagation::~BackPropagation(){

}
