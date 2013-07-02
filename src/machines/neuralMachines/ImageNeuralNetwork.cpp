#ifndef __IMAGENEURALNETWORK_HPP__
#define __IMAGENEURALNETWORK_HPP__
/*!
 * \file ImageNeuralNetwork.hpp
 * Header of the ImageNeuralNetwork class.
 * \author Luc Mioulet
 */

#include "NeuralNetwork.hpp"
#include "layers/Layer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/LayerTanh.hpp"
#include "layers/LayerSoftMax.hpp"
#include "layers/LayerSigmoid.hpp"
#include "connections/Connection.hpp"
#include "../../General.hpp"
#include "../../Clonable.hpp"
#include "../../dataset/FeatureVector.hpp"
#include "../../dataset/ErrorVector.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>

/*!
 * \class ImageNeuralNetwork
 * Description
 */
class ImageNeuralNetwork: public NeuralNetwork, public Clonable {
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
	/* Input image scaling factor*/
	realv scale;
	/* Frame size */
	int frameSize;
	/* Pixels between each frame*/
	int interFrameSpace;

public:

	/*!
	 * Default constructor.
	 */
	ImageNeuralNetwork();

	/*!
	 * Parameter constructor.
	 * \param _hidden Hidden layers.
	 * \param _connections Connections between layers.
	 * \param _forward Controlling the sequence processing of the machine.
	 * \param _name Machine name.
	 */
	ImageNeuralNetwork(std::vector<LayerPtr> _hidden, std::vector<ConnectionPtr> _connections, bool _forward = true, std::string _name = "neural_network", realv _scale = 1.0, uint frameSize=1,uint interFrameSpace=1);

	/*!
	 * Copy constructor.
	 * \param _cnn Neural network to copy.
	 */
	ImageNeuralNetwork(const ImageNeuralNetwork& _cnn);

	/*!
	 * Clone an instance of a neural network.
	 * \return A pointer to the clone.
	 */
	ImageNeuralNetwork* clone() const;

	void print(std::ostream& _os) const;

	/*!
	 * Destructor.
	 */
	~ImageNeuralNetwork();

	/*!
	 * Output file stream.
	 * \param ofs Output file stream.
	 * \param c Connection.
	 * \return Output file stream.
	 */
	friend std::ofstream& operator<<(std::ofstream& ofs, const ImageNeuralNetwork& c);

	/*!
	 * Input file stream.
	 * \param ifs Input file stream.
	 * \param c Connection.
	 * \return Input file stream.
	 */
	friend std::ifstream& operator>>(std::ifstream& ifs, ImageNeuralNetwork& c);

};

typedef boost::shared_ptr<NeuralNetwork> NeuralNetworkPtr;

#endif
