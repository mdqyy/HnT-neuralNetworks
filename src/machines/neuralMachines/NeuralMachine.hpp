#ifndef __NEURALMACHINE_HPP__
#define __NEURALMACHINE_HPP__
/*!
 * \file NeuralMachine.hpp
 * Header of the NeuralMachine class.
 * \author Luc Mioulet
 */

#include "../Machine.hpp"

/*!
 * \class NeuralMachine
 * Description
 */
class NeuralMachine: public Machine {
private:

protected:

public:

	/*!
	 * Default constructor.
	 */
	NeuralMachine();

	/*!
	 * Parameter constructor.
	 */
	NeuralMachine(std::string _name);

	/*!
	 * Forward a sequence.
	 * \param _sequence Sequence to pass forward.
	 */
	virtual void forwardSequence(std::vector<FeatureVector> _sequence)=0;

	/*!
	 * Forward a sample of a sequence.
	 * \param _sample Feature vector to pass forward.
	 */
	virtual void forward(FeatureVector _sample)=0;

	/*!
	 * Print data concerning the object.
	 * \param _os Output file stream.
	 */
	virtual void print(std::ostream& _os) const =0;

	/*!
	 * Destructor.
	 */
	virtual ~NeuralMachine();

};

#endif
