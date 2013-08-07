#ifndef MECTCTRAINER_HPP_
#define MECTCTRAINER_HPP_

/*!
 * \file MECTCTrainer.hpp
 * Header of the MECTCTrainer class.
 * \author Luc Mioulet
 */

#include "../../../machines/neuralMachines/layers/LayerCTC.hpp"
#include "../../../dataset/SequenceClassDataset.hpp"
#include "../../../dataset/ImageDataset.hpp"
#include "LearningParams.hpp"
#include <vector>
#include <opencv/cv.h>
#include "../../../machines/neuralMachines/MixedEnsembles.hpp"
#include <iostream>
#include <sstream>
#include <ostream>
#include <stdio.h>
#include "../../../dataset/ValueVector.hpp"
#include "../../../dataset/ErrorVector.hpp"
#include "../../../machines/neuralMachines/layers/Layer.hpp"

class MECTCTrainer {
private:
	/*!
	 * Calculate the required time for processing a target signal.
	 * Basically is is the length of the target signal.
	 * If two following letters in a word are the same the required length is increased by one to force the presence of a no information label.
	 * \param _targetSignal The target signal.
	 */
	uint calculateRequiredTime(std::vector<int> _targetSignal) const;

	/*!
	 * Process forward variables.
	 * \param _outputSignals Output signals from the layer.
	 * \param _targetSequence Target sequence.
	 * \return The forward variables.
	 */
	std::vector<ValueVector> processForwardVariables(std::vector<FeatureVector> _outputSignals, std::vector<int> _targetSequence);

	/*!
	 * Process backward variables.
	 * \param _outputSignals Output signals from the layer.
	 * \param _targetSequence Target sequence.
	 * \return The backward variables.
	 */
	std::vector<ValueVector> processBackwardVariables(std::vector<FeatureVector> _outputSignals, std::vector<int> _targetSequence);

	/*!
	 * Determine maximum label reached at time t given a target sequence.
	 * \param _t Time step.
	 * \param _targetSequenceSize Target sequence size.
	 * \return Maximum label reached.
	 */
	uint determineMaxLabel(uint _t, uint _targetSequenceSize);

	/*!
	 * Determine minimum start label at time t given a target sequence.
	 * \param _t Time step.
	 * \param _targetSequenceSize Target sequence size.
	 * \param _requiredSegments Required segment length.
	 * \param _targetSequenceSize Target sequence size.
	 * \return Minimum starting label.
	 */
	uint determineMinLabel(uint _t, uint _outputSignalsSize, uint _requiredSegments, uint _targetSequenceSize);

	/*!
	 * Process the forgery target of the CTC layer.
	 * \param _targetSignal Target signal.
	 * \param _outputSignals CTC output.
	 * \param _forwardVariables Forward variables.
	 * \param _backwardVariables Backward variables.
	 * \return Derivatives for CTC weight correction.
	 */
	std::vector<ErrorVector> processDerivatives(std::vector<int> _targetSignal, std::vector<FeatureVector> _outputSignals,
			std::vector<ValueVector> _forwardVariables, std::vector<ValueVector> _backwardVariables);

	/*!
	 * Process the Q value linked to the normalized C and D values.
	 *
	 */
	std::vector<realv> processQ() const;

	/*!
	 * Find unique elements in a label vector.
	 * \param Target signal.
	 * \return Vector of unique signals.
	 */
	std::vector<int> findUniqueElements(std::vector<int> _targetSignal);

	ErrorVector calculateDeltas(LayerPtr _layer, ValueVector _derivatives, ErrorVector _previousLayerDelta);

protected:
	/*!The mixed ensemble machine */
	MixedEnsembles& machine;

	/*! The CTC layer */
	LayerCTC* ctcLayer;

	/*! Training dataset */
	ImageDataset& trainingData;

	/*! Sequence dataset */
	SequenceClassDataset& trainingSequences;

	/*! Validation dataset */
	ImageDataset& validationData;

	/*! Validation sequence dataset */
	SequenceClassDataset& validationSequences;

	/* Learning parameters */
	LearningParams params;

	/*! C normalizing values produced during forward pass */
	std::vector<realv> normalizeC;

	/*! D normalizing values produced during backward pass */
	std::vector<realv> normalizeD;

	std::ostream& log;

public:
	/*!
	 * CTC layer trainer constructor.
	 * \param _ctcLayer The CTC layer to be trained.
	 * \param _data The training data.
	 * \param _featureMask Feature mask.
	 * \param _indexMask Sample index mask.
	 */
	MECTCTrainer(MixedEnsembles& _machine, ImageDataset& _trainingData, SequenceClassDataset& _trainingSequences, ImageDataset& _validationData, SequenceClassDataset& _validationSequences, LearningParams _params, std::ostream& _log);

	/*!
	 * Train the layer.
	 */
	void train();

	/*!
	 * Train the layer on one iteration.
	 */
	void trainOneIteration();

	/*!
	 * Train the layer on one sample.
	 * \param _image Input image.
	 * \param _target Target sequence.
	 */
	void trainOneSample(cv::Mat _image, std::vector<int> _targetSignal);

	/*!
	 * Backward sequence.
	 * \param _derivatives Derivatives.
	 */
	void backwardSequence(std::vector<ErrorVector> _derivatives);

	/*!
	 * Update connection of the CTC input weight matrix.
	 * \param _connection Connection input of the weight matrix.
	 * \param _deltas Derivatives.
	 */
	void updateConnection(Connection* _connection, ErrorVector _deltas);

	/*!
	 * Validate against a reference dataset.
	 */
	void validateIteration();

	/*!
	 * Used to define the index order call of the different sequences during learning.
	 * \param _numSequences Number of sequences in the data.
	 * \return Vector of unsigned integers.
	 */
	std::vector<uint> defineIndexOrderSelection(uint _numSequences);


	/*!
	 * Destructor.
	 */
	~MECTCTrainer();
};

#endif /* MECTCTRAINER_H_ */
