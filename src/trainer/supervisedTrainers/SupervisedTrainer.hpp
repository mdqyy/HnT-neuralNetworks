#ifndef __SUPERVISEDTRAINER_HPP__
#define __SUPERVISEDTRAINER_HPP__
/*!
 * \file SupervisedTrainer.hpp
 * Header of the SupervisedTrainer class.
 * \author Luc Mioulet
 */

#include "../Trainer.hpp"
#include "../../dataset/supervised/SupervisedDataset.hpp"

/*!
 * \class SupervisedTrainer
 * Abstract class for training supervised machines.
 */
class SupervisedTrainer : public Trainer{
 private :

 protected:
  SupervisedDataset& trainData;

 public:

  /*!
   * Parameter constructor.
   * \param _machine Machine to train.
   * \param _data Supervised dataset used for learning.
   * \param _featureMask Feature mask.
   * \param _indexMask Sample index mask.
   */
  SupervisedTrainer(Machine& _machine, SupervisedDataset& _data, Mask& _featureMask, Mask& _indexMask);
  
  /*!
   * Train the machine.
   */
  virtual void train() =0;    

  /*!
   * Destructor.
   */
  ~SupervisedTrainer();

};


#endif
