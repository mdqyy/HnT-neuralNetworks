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
	do{
		i++;
		trainOneIteration();
		params.setLearningRate(params.getLearningRate()*params.getLearningRateDecrease());
		params.setErrorToFirst(params.getErrorToFirst()*params.getErrorToFirstIncrease());
	}while(i<params.getMaxIterations());
}

/*!
 * Thread function forwarding data and extracting error.
 */
ErrorVector calculateDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta){
	ErrorVector delta = ErrorVector(_layer->getNumUnits());
	for(uint i=0;i<delta.getLength();i++){
		delta[i]=_derivatives[i]*_layer->errorWeighting(_previousLayerDelta,_layer->getOutputConnection()->getWeightsFromNeuron(i));
	}
	return delta;
}

ErrorVector calculateOutputDeltas(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives){
	ErrorVector delta = ErrorVector(_layer->getNumUnits());
	for(uint i=0;i<_target.getLength();i++){
		delta[i] = _derivatives[i]*(_target[i]-_layer->getOutputSignal()[i]);  // error calculation if output layer
	}
	return delta;
}

void updateConnection(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate){
	Mat weights = _connection->getWeights();
	for(int i=0;i<weights.rows;i++){
		for(int j=0;j<weights.cols;j++){
			weights.at<realv>(i,j)=weights.at<realv>(i,j)+_learningRate*_deltas[i]*_connection->getInputLayer()->getOutputSignal()[j];
		}
	}
	_connection->setWeights(weights);
}

void backward(NeuralNetworkPtr _neuralNet,FeatureVector _target, realv _learningRate){
	vector<ConnectionPtr> connections = _neuralNet->getConnections();
	vector<LayerPtr> layers = _neuralNet->getHiddenLayers();
	vector<ErrorVector> deltas = vector<ErrorVector>();/* pushed in inversed order so be careful */
	for(uint i = layers.size() -1 ; i > 0; i--){
		ValueVector derivatives = layers[i]->getDerivatives();
		if(i == layers.size()-1){
			deltas.push_back(calculateOutputDeltas(layers[i], _target, derivatives));
		}
		else{
			deltas.push_back(calculateDeltas(layers[i], _target, derivatives, deltas[deltas.size()-1]));
		}
	}
	for(uint i = 0; i < connections.size(); i++){
		updateConnection(connections[i], deltas[connections.size()-i-1],_learningRate);
	}
}
void forwardNetworksThread(vector<NeuralNetworkPtr> _neuralNets, uint _k, vector<MSEMeasurer>* _mses, FeatureVector _fv){
	_neuralNets[_k]->forward(_fv);
	(*_mses)[_k].totalError(_neuralNets[_k]->getOutputSignal(),_fv);
}

void backwardNetworksThread(vector<NeuralNetworkPtr> _neuralNets, uint _k, FeatureVector _target, realv _learningRate, realv _minError, vector<MSEMeasurer>* _mses, realv _similarity,vector<bool>* _trained){
	if(_minError/ (*_mses)[_k].getError()>= _similarity){
		(*_trained)[_k]=true;
		backward(_neuralNets[_k], _target, _learningRate);
	}
	else{
		backward(_neuralNets[_k],FeatureVector(_target.getLength()), _learningRate);
	}
}

void forwardBackwardNetworksThread(vector<NeuralNetworkPtr> _neuralNets, uint _k, vector<MSEMeasurer>* _mses, FeatureVector _fv, realv _lr, realv* _error,uint _numSamples){
	RNG random;
	random.next();
	_neuralNets[_k]->forward(_fv);
	(*_mses)[_k].totalError(_neuralNets[_k]->getOutputSignal(),_fv);
	backward(_neuralNets[_k],_fv, _lr*random.gaussian(5.0));
	*_error += ((*_mses)[_k].getError())/((realv)_neuralNets.size());
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
				threads.push_back(new boost::thread(forwardBackwardNetworksThread,neuralNets,k,&mses,fv,params.getLearningRate(),&netError, data.getNumSamples()));
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
			FeatureVector fv= data[index][j];
			for(uint k=0;k<neuralNets.size();k++){
				threads.push_back(new boost::thread(forwardNetworksThread,neuralNets,k,&mses,fv));
			}
			for(uint k=0; k<neuralNets.size();k++){
				threads[k]->join();
				delete threads[k];
			}
			for(uint k=0;k<neuralNets.size();k++){
				if(mses[k].getError()<minError){
					minError=mses[k].getError();
				}
			}
			realv similarity=params.getErrorToFirst();
			uint numberTrained=0;
			vector<bool> trained(neuralNets.size(),false);
			vector<boost::thread *> threadsBackward;
			for(uint k=0;k<neuralNets.size();k++){
				threadsBackward.push_back(new boost::thread(backwardNetworksThread,neuralNets,k,fv,params.getLearningRate(), minError, &mses,similarity,&trained));
			}
			for(uint k=0; k<neuralNets.size();k++){
				threadsBackward[k]->join();
				delete threadsBackward[k];
			}
			for(uint k=0; k<neuralNets.size();k++){
				if(trained[k]==true){
					sampleError+= mses[k].getError();
					histogramOfTrainees[k] = histogramOfTrainees[k]+1;
					for(uint l=0;l<neuralNets.size();l++){
						if(l!=k && trained[l]){
							correlatedTraining[l][k] +=1;
							correlatedTraining[k][l] +=1;
						}
					}
					numberTrained++;
					averageNumberTrained +=1.0;
				}
			}
			/*if(minError/mses[k].getError()>=similarity){
				trained[k]=true;
				backward(neuralNets[k], data[index][j], params.getLearningRate());
				sampleError+= mses[k].getError();
				histogramOfTrainees[k] = histogramOfTrainees[k]+1;
				for(uint l=0;l<neuralNets.size();l++){
					if(l!=k && trained[l]){
						correlatedTraining[l][k] +=1;
						correlatedTraining[k][l] +=1;
					}
				}
				numberTrained++;
				averageNumberTrained +=1.0;
			}
			else{
				backward(neuralNets[k],FeatureVector(data[index][j].getLength()), params.getLearningRate());
			}*/
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
			cout << correlatedTraining[k][l]/((realv)data.getNumSamples())<<" ";
		}
		cout << endl;
	}
	averageNumberTrained/=((realv)data.getNumSamples());
	cout << "Iteration error " << (iterationError/((realv)data.getNumSequences())) <<", average number of networks trained "<< averageNumberTrained << endl;
}

PopulationBP::~PopulationBP(){

}
