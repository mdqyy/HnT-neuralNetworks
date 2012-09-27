/*!
 * \file TextUtilities.cpp
 * Body of the text utilities.
 * \author Luc Mioulet
 */

#include "TextUtilities.hpp"

using namespace std;

/*!
 * Extract a sequence label from a string.
 * Basically cuts the string into characters.
 */
vector<string> extractLabelSequence(string _label){
	vector<string> labels = vector<string>();
	for(int i=0;i<_label.size();i++){
		stringstream ss;
		ss << _label[i];
		labels.push_back(ss.str());
	}
	return labels;
}
