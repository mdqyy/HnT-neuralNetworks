/*!
 * \file PopulationBP.cpp
 * Body of the PopulationBP class.
 * \author Luc Mioulet
 */

#include "PopulationBP.hpp"

using namespace cv;
using namespace std;

PopulationBP::PopulationBP(PBDNN& _population, SupervisedDataset& _data, PopulationBPParams& _params, Mask& _featureMask, Mask& _indexMask) : SupervisedTrainer(_population, _data, _featureMask, _indexMask), population(_population), params(_params){

}

void PopulationBP::train(){
  uint i=0;
  preTrain();
  do{
    i++;
    trainOneIteration();
    params.setLearningRate(params.getLearningRate()*params.getLearningRateDecrease());
  }while(i<params.getMaxIterations());
}

/*!
 * Thread function forwarding data and extracting error.
 */
void forwardNetworksThread(vector<NeuralNetworkPtr> neuralNets, uint j, uint k, vector<MSEMeasurer>* mses, FeatureVector fv){
  neuralNets[k]->forward(fv);
  (*mses)[k].totalError(neuralNets[k]->getOutputSignal(),fv);
}

void forwardBackwardNetworksThread(vector<NeuralNetworkPtr> neuralNets, uint j, uint k, vector<MSEMeasurer>* mses, FeatureVector fv, realv lr, realv *error,uint numSamples){
  RNG random;
  random.next();
  neuralNets[k]->forward(fv);
  (*mses)[k].totalError(neuralNets[k]->getOutputSignal(),fv);
  neuralNets[k]->backward(fv, lr*random.gaussian(1.0));
  *error += ((*mses)[k].getError())/((realv)neuralNets.size());
}

void PopulationBP::preTrain(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  uint index=0;

  vector<FeatureVector> errors;
  vector<NeuralNetworkPtr> neuralNets=population.getPopulation();
  vector<MSEMeasurer> mses(neuralNets.size());
  realv iterationError=0;
  for(uint i=0; i<data.getNumSequences();i++){
    index=indexOrderSelection[i];
    realv seqError=0;
    for(uint j=0;j<data[index].size();j++){
      list<realv> scoreList;
      list<int> indexList;
      vector<boost::thread *> threads;
      realv netError = 0;
      for(uint k=0;k<neuralNets.size();k++){
	FeatureVector fv= data[index][j];
	threads.push_back(new boost::thread(forwardBackwardNetworksThread,neuralNets,j,k,&mses,fv,params.getLearningRate(),&netError, data.getNumSamples()));
      }
      for(uint k=0; k<neuralNets.size();k++){
	threads[k]->join();
	delete threads[k];
      }
      seqError+=netError/((realv)data[index].size());
    }
    iterationError += seqError/((realv)data.getNumSequences());
  }
  cout << "Pretrain error " << iterationError << endl;
}

void PopulationBP::trainOneIteration(){
  vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
  uint index=0;

  vector<FeatureVector> errors;
  vector<NeuralNetworkPtr> neuralNets=population.getPopulation();
  vector<MSEMeasurer> mses(neuralNets.size());
  vector<uint> histogramOfTrainees(neuralNets.size());
  vector< vector<uint> > correlatedTraining;
  for(uint k=0; k<neuralNets.size();k++){
    mses[k] = MSEMeasurer();
    histogramOfTrainees[k] = 0;
    correlatedTraining.push_back(vector<uint>(neuralNets.size()));
    for(uint l=0; l<neuralNets.size();l++){
      correlatedTraining[k][l]=0;
    }
  }
  realv iterationError=0;
  realv averageNumberTrained=0;
  for(uint i=0; i<data.getNumSequences();i++){
    realv seqError = 0;
    index=indexOrderSelection[i];
    for(uint j=0;j<data[index].size();j++){
      realv sampleError = 0;
      realv minError=10e+9;
      list<realv> scoreList;
      list<int> indexList;
      vector<boost::thread *> threads;
      for(uint k=0;k<neuralNets.size();k++){
	FeatureVector fv= data[index][j];
	threads.push_back(new boost::thread(forwardNetworksThread,neuralNets,j,k,&mses,fv));
      }
      for(uint k=0; k<neuralNets.size();k++){
	threads[k]->join();
	delete threads[k];
      }
      for(uint k=0;k<neuralNets.size();k++){
	if(mses[k].getError()<minError){
	  minError=mses[k].getError();
	  scoreList.push_front(mses[k].getError());
	  indexList.push_front(k);
	}
	else{
	  scoreList.push_back(mses[k].getError());
	  indexList.push_back(k);
	}
      }
      list<int>::iterator iterIndex=indexList.begin();
      list<realv>::iterator iterScore=scoreList.begin();
      realv errorDifference=params.getErrorToFirst();
      uint numberTrained=0;
      realv trainedError=0;
      vector<bool> trained(neuralNets.size(),false);
      while(abs(minError/(*iterScore))>errorDifference && iterScore!=scoreList.end() && numberTrained<params.getMaxTrained() && numberTrained<neuralNets.size()){
	trained[*iterIndex]=true;
	neuralNets[*iterIndex]->backward(data[index][j], params.getLearningRate());
	sampleError+= *iterScore;
	trainedError += mses[*iterIndex].getError();
	histogramOfTrainees[*iterIndex] = histogramOfTrainees[*iterIndex]+1;
	for(int l=0;l<neuralNets.size();l++){
	  if(l!=(*iterIndex) && trained[l]){
	    correlatedTraining[l][*iterIndex] +=1;
	    correlatedTraining[*iterIndex][l] +=1;
	  }
	}
	iterIndex++;
	iterScore++;
	numberTrained++;
	averageNumberTrained +=1.0;
      }
      seqError += sampleError/((realv)numberTrained);
    }
    iterationError+=seqError/((realv)data[index].size());
  }
  for(uint k=0; k<neuralNets.size();k++){
    cout << k << "\t "<< histogramOfTrainees[k] << endl;
  }
  cout << "Correlated trainings" << endl;
  for(uint k=0; k<neuralNets.size();k++){
    for(uint l=0; l<neuralNets.size();l++){
      cout << correlatedTraining[k][l]<<" ";
    }
    cout << endl;
  }
  averageNumberTrained/=((realv)data.getNumSamples());
  cout << "Iteration error " << (iterationError/((realv)data.getNumSequences())) <<", average number of networks trained "<< averageNumberTrained << endl;
}

PopulationBP::~PopulationBP(){

}
