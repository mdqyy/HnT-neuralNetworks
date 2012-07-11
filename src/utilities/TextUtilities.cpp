/*!
 * \file TextUtilities.cpp
 * Body of the text utilities.
 * \author Luc Mioulet
 */

#include "TextUtilities.hpp"

using namespace std;

vector<string> extractLabelSequence(string label){
	vector<string> labels = vector<string>();
	cout << label << endl;
	for(int i=0;i<label.size();i++){
		stringstream ss;
		ss << label[i];
		labels.push_back(ss.str());
	}
	cout << endl;
	return labels;
}
