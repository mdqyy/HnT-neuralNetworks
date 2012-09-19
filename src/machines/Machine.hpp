#ifndef __MACHINE_HPP__
#define __MACHINE_HPP__
/*!
 * \file Machine.hpp
 * Header of the Machine class.
 * \author Luc Mioulet
 */

#include "../dataset/FeatureVector.hpp"
#include <string>
#include <vector>

/*!
 * \class Machine
 * Description
 */
class Machine {
private:

protected:
	/*! Machine name */
	std::string name;

public:

	/*!
	 * Default constructor.
	 */
	Machine();

	/*!
	 * Constructor.
	 * \param _name Machine name.
	 */
	Machine(std::string _name);

	/*!
	 * Get name.
	 * \return Machine name.
	 */
	std::string getName() const;

	/*!
	 * Set name
	 * \param _name New name.
	 */
	void setName(std::string _name);

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
	~Machine();

	/*!
	 * Print some machine information.
	 * \param _os Output stream.
	 * \param _m Machine.
	 * \remark Uses the overloaded function print.
	 */
	friend std::ostream& operator<<(std::ostream& _os, const Machine& _m);
};

#endif
