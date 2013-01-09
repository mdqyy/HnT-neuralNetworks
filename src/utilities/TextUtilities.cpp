/*!
 * \file TextUtilities.cpp
 * Body of the text utilities.
 * \author Luc Mioulet
 */

#include "TextUtilities.hpp"
#include <cstring>
#include <wchar.h>
#include <cstdlib>

using namespace std;

vector<string> extractLabelSequence(string _label) {
	vector<string> labels = vector<string>();
	int i = 0;
	string substring;
	while (i < _label.size()) {
		stringstream ss;
		substring = _label.substr(i, 1);
		if (substring.compare("[a-z][A-Z]") > 32) {
			substring = _label.substr(i,2);
			i = i + 2;
		} else {
			i = i + 1;
		}
		labels.push_back(substring);
	}
	return labels;
}

void addDictionaryClasses(ClassificationDataset* _dataset) {
	_dataset->addClass("a");
	_dataset->addClass("â");
	_dataset->addClass("à");
	_dataset->addClass("b");
	_dataset->addClass("c");
	_dataset->addClass("ç");
	_dataset->addClass("d");
	_dataset->addClass("e");
	_dataset->addClass("ê");
	_dataset->addClass("é");
	_dataset->addClass("è");
	_dataset->addClass("ë");
	_dataset->addClass("f");
	_dataset->addClass("g");
	_dataset->addClass("h");
	_dataset->addClass("i");
	_dataset->addClass("ï");
	_dataset->addClass("î");
	_dataset->addClass("j");
	_dataset->addClass("k");
	_dataset->addClass("l");
	_dataset->addClass("m");
	_dataset->addClass("n");
	_dataset->addClass("o");
	_dataset->addClass("ô");
	_dataset->addClass("œ");
	_dataset->addClass("p");
	_dataset->addClass("q");
	_dataset->addClass("r");
	_dataset->addClass("s");
	_dataset->addClass("t");
	_dataset->addClass("u");
	_dataset->addClass("ù");
	_dataset->addClass("ü");
	_dataset->addClass("û");
	_dataset->addClass("v");
	_dataset->addClass("w");
	_dataset->addClass("x");
	_dataset->addClass("y");
	_dataset->addClass("z");
	_dataset->addClass("A");
	_dataset->addClass("À");
	_dataset->addClass("Â");
	_dataset->addClass("B");
	_dataset->addClass("C");
	_dataset->addClass("Ç");
	_dataset->addClass("D");
	_dataset->addClass("E");
	_dataset->addClass("É");
	_dataset->addClass("È");
	_dataset->addClass("Ê");
	_dataset->addClass("Ë");
	_dataset->addClass("F");
	_dataset->addClass("G");
	_dataset->addClass("H");
	_dataset->addClass("I");
	_dataset->addClass("Î");
	_dataset->addClass("Ï");
	_dataset->addClass("J");
	_dataset->addClass("K");
	_dataset->addClass("L");
	_dataset->addClass("M");
	_dataset->addClass("N");
	_dataset->addClass("O");
	_dataset->addClass("Ô");
	_dataset->addClass("Œ");
	_dataset->addClass("P");
	_dataset->addClass("Q");
	_dataset->addClass("R");
	_dataset->addClass("S");
	_dataset->addClass("T");
	_dataset->addClass("U");
	_dataset->addClass("Ù");
	_dataset->addClass("Û");
	_dataset->addClass("Ü");
	_dataset->addClass("V");
	_dataset->addClass("W");
	_dataset->addClass("X");
	_dataset->addClass("Y");
	_dataset->addClass("Z");
	_dataset->addClass(",");
	_dataset->addClass(".");
	_dataset->addClass(":");
	_dataset->addClass("'");
	_dataset->addClass("°");
	_dataset->addClass("%");
	_dataset->addClass("-");
	_dataset->addClass("/");
	_dataset->addClass("&");
	_dataset->addClass("!");
	_dataset->addClass(")");
	_dataset->addClass("(");
	_dataset->addClass("«");
	_dataset->addClass("»");
	_dataset->addClass("\"");
	_dataset->addClass("0");
	_dataset->addClass("1");
	_dataset->addClass("2");
	_dataset->addClass("3");
	_dataset->addClass("4");
	_dataset->addClass("5");
	_dataset->addClass("6");
	_dataset->addClass("7");
	_dataset->addClass("8");
	_dataset->addClass("9");
	_dataset->addClass("²");
}
