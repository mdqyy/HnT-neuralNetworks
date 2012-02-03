#ifndef __REGRESSIONDATASET_HPP__
#define __REGRESSIONDATASET_HPP__
/*!
 * \file RegressionDataset.hpp
 * Header of the RegressionDataset class.
 * \author Luc Mioulet
 */

#include "SupervisedDataset.hpp"
#include <string>

/*!
 * \class RegressionDataset
 * Description
 */
class RegressionDataset : public SupervisedDataset{
 private :

 protected:

 public:

  /*!
   * Default constructor.
   */
  RegressionDataset(std::string _fileName);

  /*!
   * Get dataset type.
   * \return Dataset type.
   */
  virtual int getDatasetType() const;

  /*!
   * Destructor.
   */
  ~RegressionDataset();

};


#endif
