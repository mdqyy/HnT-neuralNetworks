#ifndef __NEURALNETWORK_HPP__
#define __NEURALNETWORK_HPP__
/*!
 * \file NeuralNetwork.hpp
 * Header of the NeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralMachine.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/LayerTanh.hpp"
#include "layers/LayerSoftMax.hpp"
#include "layers/LayerSigmoid.hpp"
#include "layers/LayerCTC.hpp"
#include "connections/Connection.hpp"
#include "../../General.hpp"
#include "../../Clonable.hpp"
#include "../../dataset/FeatureVector.hpp"
#include "../../dataset/ErrorVector.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>

/*!
 * \class NeuralNetwork
 * Description
 */
class NeuralNetwork: public NeuralMachine, public Clonable {
private:

protected:
	/*! Hidden layer */
	std::vector<LayerPtr> hiddenLayers;
	/*! Connections between layers */
	std::vector<ConnectionPtr> connections;
	/*! Input signal */
	FeatureVector inputSignal;
	/*! Output signal */
	FeatureVector outputSignal;
	/*! Forward or backward sequence processing. */
	bool readForward;

public:

	/*!
	 * Default constructor.
	 */
	NeuralNetwork();

	/*!
	 * Parameter constructor.
	 * \param _hidden Hidden layers.
	 * \param _connections Connections between layers.
	 * \param _forward Controlling the sequence processing of the machine.
	 * \param _name Machine name.
	 */
	NeuralNetwork(std::vector<LayerPtr> _hidden, std::vector<ConnectionPtr> _connections, bool _forward = true, std::string _name = "neural_network");

	/*!
	 * Copy constructor.
	 * \param _cnn Neural network to copy.
	 */
	NeuralNetwork(const NeuralNetwork& _cnn);

	/*!
	 * Clone an instance of a neural network.
	 * \return A pointer to the clone.
	 */
	NeuralNetwork* clone() const;

	/*!
	 * Get input layer.
	 * \return Input layer.
	 */
	Layer* getInputLayer() const;

	/*!
	 * Get hidden layers.
	 * \return Hidden layers.
	 */
	std::vector<LayerPtr> getHiddenLayers() const;

	/*!
	 * Get output layer.
	 * \return Output layer.
	 */
	Layer* getOutputLayer() const;

	/*!
	 * Get input signal.
	 * \return Input signal.
	 */
	FeatureVector getInputSignal() const;

	/*!
	 * Get output signal.
	 * \return Output signal.
	 */
	FeatureVector getOutputSignal() const;

	/*!
	 * Get Connections in the neural network.
	 * \return Connections.
	 */
	std::vector<ConnectionPtr> getConnections() const;

	/*!
	 * Get the forward
	 * \return True if the network reads the sequence in a forward manner. False if it reads it backwards.
	 */
	bool isForward() const;

	/*!
	 * Set hidden layers.
	 * \param _hidden Hidden layers.
	 */
	void setHiddenLayers(std::vector<LayerPtr> _hidden);

	/*!
	 * Forward a sequence.
	 * \param _sequence Sequence of feature vectors.
	 */
	void forwardSequence(std::vector<FeatureVector> _sequence);

	/*!
	 * Forward a feature vector.
	 * \param signal Input feature vector.
	 */
	void forward(FeatureVector _signal);

	/*!
	 * Suppress the last layer, the next to last will replace it.
	 */
	void suppressLastLayer();

	/*!
	 * Add a new Layer on top of the others.
	 */
	void addLayer(LayerPtr lptr);

	/*!
	 * Backward an error vector.
	 * \param _target Neural network target.
	 * \param _learningRate Weight change rate.
	 */
	// void backward(FeatureVector _target, realv _learningRate);
	/*!
	 * Print data concerning the object.
	 * \param _os Output file stream.
	 */
	void print(std::ostream& _os) const;

	/*!
	 * Destructor.
	 */
	~NeuralNetwork();

	/*!
	 * Output file stream.
	 * \param ofs Output file stream.
	 * \param c Connection.
	 * \return Output file stream.
	 */
	friend std::ofstream& operator<<(std::ofstream& ofs, const NeuralNetwork& c);

	/*!
	 * Input file stream.
	 * \param ifs Input file stream.
	 * \param c Connection.
	 * \return Input file stream.
	 */
	friend std::ifstream& operator>>(std::ifstream& ifs, NeuralNetwork& c);

};

typedef boost::shared_ptr<NeuralNetwork> NeuralNetworkPtr;

#endif
