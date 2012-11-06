#ifndef __SUPERVISEDTRAINER_HPP__
#define __SUPERVISEDTRAINER_HPP__
/*!
 * \file SupervisedTrainer.hpp
 * Header of the SupervisedTrainer class.
 * \author Luc Mioulet
 */

#include "../Trainer.hpp"
#include "../../dataset/supervised/SupervisedDataset.hpp"
#include <opencv/cv.h>
#include <vector>

/*!
 * \class SupervisedTrainer
 * Abstract class for training supervised machines.
 */
class SupervisedTrainer: public Trainer {
private:

protected:
	/*! Supervised training data */
	SupervisedDataset& trainData;

public:

	/*!
	 * Parameter constructor.
	 * \param _machine Machine to train.
	 * \param _data Supervised dataset used for learning.
	 * \param _featureMask Feature mask.
	 * \param _indexMask Sample index mask.
	 */
	SupervisedTrainer(Machine& _machine, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask);

	/*!
	 * Used to define the index order call of the different sequences during learning.
	 * \param _numSequences Number of sequences in the data.
	 * \return Vector of unsigned integers.
	 */
	std::vector<uint> defineIndexOrderSelection(uint _numSequences);

	/*!
	 * Train the machine.
	 */
	virtual void train() =0;

	/*!
	 * Destructor.
	 */
	~SupervisedTrainer();

};

#endif
