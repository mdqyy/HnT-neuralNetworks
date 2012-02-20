#ifndef __CROSSVALIDATIONPARAMS_HPP__
#define __CROSSVALIDATIONPARAMS_HPP__
/*!
 * \file CrossValidationParams.hpp
 * Header of the CrossValidationParams class.
 * \author Luc Mioulet
 */
#include <opencv/cv.h>

#define NOCV 0
#define KFOLDCV 1
#define LOOCV 2
#define REPEATEDRANDOMSUBSAMPLING 3

/*!
 * \class CrossValidationParams
 * Parametrisation of the cross validation.
 */
class CrossValidationParams {
 private :

 protected:
  bool doCV;
  int crossValidationType;
  uint numberOfFolds;

 public:

  /*!
   * Default constructor. 
   * Does not activate cross validation.
   */
  CrossValidationParams();

  /*!
   * Parameter constructor.
   * \param _crossValidationType Cross validation type. 
   * \param _numberOfFolds Number of folds (not for LOOCV).
   */
  CrossValidationParams(int _crossValidationType, int _numberOfFolds=10);
  
  /*!
   * Use or not cross validation.
   * \return Should cross validation be used.
   */
  bool doCrossValidation() const;
  
  /*!
   * Get cross validation type.
   * \return Cross validation type.
   */
  int getCrossValidationType() const;

  /*!
   * Get number of folds.
   * \return Number of folds
   */
  uint getNumberOfFolds() const;

  /*!
   * Destructor.
   */
  ~CrossValidationParams();

};


#endif
