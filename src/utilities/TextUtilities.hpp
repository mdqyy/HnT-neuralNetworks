#ifndef __TEXTUTILITIES_HPP__
#define __TEXTUTILITIES_HPP__
/*!
 * \file TextUtilities.hpp
 * Header of the text utilities.
 * \author Luc Mioulet
 */

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "../dataset/supervised/ClassificationDataset.hpp"

/*!
 * Extract a sequence label from a string.
 * Basically cuts the string into characters.
 * \param _label Basic string to be cut into separate labels.
 * \return The vector of labels.
 */
std::vector<std::string> extractLabelSequence(std::string _label);

/*!
 * Add dictionnary classes using a french dictionnary.
 * \param _dataset The classification dataset.
 */
void addDictionaryClasses(ClassificationDataset* _dataset);

std::string &ltrim(std::string &s);

// trim from end
std::string &rtrim(std::string &s);

// trim from both ends
std::string &trim(std::string &s);


#endif
