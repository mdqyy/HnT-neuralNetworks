/*!
 * \file PopulationClusterBP.cpp
 * Body of the PopulationClusterBP class.
 * \author Luc Mioulet
 */

#include "PopulationClusterBP.hpp"
#include "../../errorMeasurers/AEMeasurer.hpp"

using namespace cv;
using namespace std;

PopulationClusterBP::PopulationClusterBP(PBDNN& _population, RegressionDataset& _data, PopulationBPParams& _params, Mask& _featureMask, Mask& _indexMask) : SupervisedTrainer(_population, _data, _featureMask, _indexMask), population(_population), params(_params), regData(_data){

}

void PopulationClusterBP::train(){
	uint i=0;
	do{
		i++;
		cout << "Iteration" << i <<endl;
		trainOneIteration();
		params.setLearningRate(params.getLearningRate()*params.getLearningRateDecrease());
		params.setErrorToFirst(params.getErrorToFirst()*params.getErrorToFirstIncrease());
	}while(i<params.getMaxIterations());
}



/*!
 * Thread function forwarding data and extracting error.
 */
ErrorVector calculateDeltas2(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives, ErrorVector _previousLayerDelta){
	ErrorVector delta = ErrorVector(_layer->getNumUnits());
	for(uint i=0;i<delta.getLength();i++){
		delta[i]=_derivatives[i]*_layer->errorWeighting(_previousLayerDelta,_layer->getOutputConnection()->getWeightsFromNeuron(i));
	}
	return delta;
}

ErrorVector calculateOutputDeltas2(LayerPtr _layer, FeatureVector _target, ValueVector _derivatives){
	ErrorVector delta = ErrorVector(_layer->getNumUnits());
	for(uint i=0;i<_target.getLength();i++){
		delta[i] = _derivatives[i]*(_target[i]-_layer->getOutputSignal()[i]);  // error calculation if output layer
	}
	return delta;
}

void updateConnection2(ConnectionPtr _connection, ErrorVector _deltas, realv _learningRate){
	Mat weights = _connection->getWeights();
	for(int i=0;i<weights.rows;i++){
		for(int j=0;j<weights.cols;j++){
			weights.at<realv>(i,j)=weights.at<realv>(i,j)+_learningRate*_deltas[i]*_connection->getInputLayer()->getOutputSignal()[j];
		}
	}
	_connection->setWeights(weights);
}

void backward2(NeuralNetworkPtr _neuralNet,FeatureVector _target, realv _learningRate){
	vector<ConnectionPtr> connections = _neuralNet->getConnections();
	vector<LayerPtr> layers = _neuralNet->getHiddenLayers();
	vector<ErrorVector> deltas = vector<ErrorVector>();/* pushed in inversed order so be careful */
	for(uint i = layers.size() -1 ; i > 0; i--){
		ValueVector derivatives = layers[i]->getDerivatives();
		if(i == layers.size()-1){
			deltas.push_back(calculateOutputDeltas2(layers[i], _target, derivatives));
		}
		else{
			deltas.push_back(calculateDeltas2(layers[i], _target, derivatives, deltas[deltas.size()-1]));
		}
	}
	for(uint i = 0; i < connections.size(); i++){
		updateConnection2(connections[i], deltas[connections.size()-i-1],_learningRate);
	}
}
void forwardNetworksThread2(vector<NeuralNetworkPtr> _neuralNets, uint _k, vector<AEMeasurer>* _mses, FeatureVector _fv){
	_neuralNets[_k]->forward(_fv);
	(*_mses)[_k].totalError(_neuralNets[_k]->getOutputSignal(),_fv);
}

void backwardNetworksThread(vector<NeuralNetworkPtr> _neuralNets, uint _k, FeatureVector _target,realv _learningRate, realv _minError, vector<AEMeasurer>* _mses, realv _similarity,vector<bool>* _trained){
	if(_minError/ (*_mses)[_k].getError()>= _similarity){
		(*_trained)[_k]=true;
		backward2(_neuralNets[_k], _target, _learningRate);
	}
}

void forwardBackwardNetworksThread(vector<NeuralNetworkPtr> _neuralNets, uint _k, vector<AEMeasurer>* _mses, FeatureVector _fv, realv _lr, realv* _error,uint _numSamples){
	_neuralNets[_k]->forward(_fv);
	(*_mses)[_k].totalError(_neuralNets[_k]->getOutputSignal(),_fv);
	backward2(_neuralNets[_k],_fv, _lr);
	*_error += ((*_mses)[_k].getError())/((realv)_neuralNets.size());
}

void PopulationClusterBP::trainOneIteration(){
	vector<uint> indexOrderSelection=defineIndexOrderSelection(data.getNumSequences());
	uint index=0;
	uint bestNetwork=0;
	DiversityMeasurer diversity(population,regData);
	vector<FeatureVector> errors;
	vector<NeuralNetworkPtr> neuralNets=population.getPopulation();
	vector<vector<Mat> > tempWeights;
	AEMeasurer mae;
	vector<uint> histogramOfTrainees(neuralNets.size());
	vector< vector<uint> > correlatedTraining;
	vector <vector< pair<int,int> > > learningAffectations = vector <vector< pair<int,int> > >();
	for(uint k=0; k<neuralNets.size();k++){
		histogramOfTrainees[k] = 0;
		correlatedTraining.push_back(vector<uint>(neuralNets.size()));
		learningAffectations.push_back(vector<pair<int,int> >());
		tempWeights.push_back(vector<Mat>());
		vector<ConnectionPtr> connections =neuralNets[k]->getConnections();
		for(uint i = 0; i < connections.size(); i++){
			tempWeights[k].push_back(connections[i]->getWeights().clone());
		}
		for(uint l=0; l<neuralNets.size();l++){
			correlatedTraining[k][l]=0;
		}
	}
	for(uint i=0; i<data.getNumSequences();i++){
		index=indexOrderSelection[i];
		for(uint j=0;j<data[index].size();j++){
			realv minError=10e+9;
			FeatureVector fv= data[index][j];
			bestNetwork = 0;
			for(uint k=0;k<neuralNets.size();k++){
				neuralNets[k]->forward(fv);
				mae.processErrors(neuralNets[k]->getOutputSignal(),regData.getTargetSample(index,j));
				if(mae.getError()<minError){
					minError=mae.getError();
					bestNetwork = k;
				}
			}
			learningAffectations[bestNetwork].push_back(pair<int,int>(index,j));
		}
	}
	uint timesTrained = data.getNumSamples();
	for(uint k=0; k<neuralNets.size();k++){
		if(learningAffectations[k].size()<timesTrained){
			timesTrained = learningAffectations[k].size();
		}
	}
	FeatureVector blackTarget = FeatureVector(regData.getTargetSample(0, 0));
	for(uint k=0;k<neuralNets.size();k++){
		for(uint i=0;i<timesTrained;i++){
			/* forward backward good sample */
			pair<int, int> index = learningAffectations[k][i];
			neuralNets[k]->forward(regData[index.first][index.second]);
			backward2(neuralNets[k], regData.getTargetSample(index.first, index.second), params.getLearningRate());
			/* forward backward bad random sample */
			RNG randomK;
			randomK.next();
			uint randK=0;
			do{
				randK = randomK.uniform(0,neuralNets.size());
			}while(randK==k);
			index = learningAffectations[randK][i];
			neuralNets[k]->forward(regData[index.first][index.second]);
			backward2(neuralNets[k], blackTarget, params.getLearningRate());
		}
	}
	cout << "network  | \t Global error \t | \t timesSelected" << endl;
	for(uint i=0;i<neuralNets.size();i++){
		RegressionMeasurer regMeasurer = RegressionMeasurer(*(neuralNets[i].get()),regData,mae);
		regMeasurer.measurePerformance();
		cout << i << " \t | \t " << regMeasurer.getTotalError() <<"\t | \t" << learningAffectations[i].size() <<endl;
	}
	diversity.measurePerformance();
	cout << "Diversity : " << endl << diversity.getDisagreementMatrix() << endl;
	cout << "Diversity scalar: " << endl << diversity.getDisagreementScalar() << endl;
}

PopulationClusterBP::~PopulationClusterBP(){

}
