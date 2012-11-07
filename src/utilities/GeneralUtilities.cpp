/*!
 * \file GeneralUtilities.cpp
 * Body of the general utilities.
 * \author Luc Mioulet
 */

#include "GeneralUtilities.hpp"

using namespace std;

string helper(string _name, string _aim, vector<string> _arguments) {
	ostringstream os;
	os << _name << endl << "Description : " << _aim << endl << "Arguments : " << endl;
	for (int i = 0; i < _arguments.size(); i++) {
		os << " \t - " << _arguments[i] << endl;
	}
	return os.str();
}

