#ifndef __GENERALUTILITIES_HPP__
#define __GENERALUTILITIES_HPP__
/*!
 * \file GeneralUtilities.hpp
 * Header of the general utilities.
 * \author Luc Mioulet
 */

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "../dataset/ValueVector.hpp"

/*!
 * Help prompt when starting a program 
 * \param _name Name of the program.
 * \param _aim Global description.
 * \param _arguments Input arguments.
 * \return Prompt output.
 */

std::string helper(std::string _name, std::string _aim, std::vector<std::string> _arguments);

/*!
 * Concatenate a vector of vectors.
 * \param _vecs Vector of vectors.
 * \return The concatenated vector.
 */
ValueVector concatenate(std::vector<ValueVector> _vecs);

#endif
