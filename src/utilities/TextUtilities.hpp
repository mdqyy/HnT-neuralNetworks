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

/*!
 * Extract a sequence label from a string.
 * Basically cuts the string into characters.
 * \param _label Basic string to be cut into separate labels.
 * \return The vector of labels.
 */
std::vector<std::string> extractLabelSequence(std::string _label);

#endif
