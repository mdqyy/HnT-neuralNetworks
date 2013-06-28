#ifndef __GENERAL_HPP__
#define __GENERAL_HPP__
/*!
 * \file General.hpp
 * Header containing used typedefs.
 * \author Luc Mioulet
 */

/*! \todo decide in cmake */
#define REAL_DOUBLE 1

#include <vector>

#ifdef REAL_DOUBLE
typedef double realv;
#else
typedef float realv;
#endif

typedef std::vector< std::vector<realv> > weights;

#endif
