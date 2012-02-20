#ifndef __TRAINER_HPP__
#define __TRAINER_HPP__
/*!
 * \file Trainer.hpp
 * Header of the Trainer class.
 * \author Luc Mioulet
 */

#include "../dataset/Dataset.hpp"
#include "../dataset/Mask.hpp"
#include "../machines/Machine.hpp"

/*!
 * \class Trainer
 * Abstract class for training machines.
 */
class Trainer {
 private :

 protected:
  /*! Machine to be trained */
  Machine& machine;
  /*! Dataset used for training */
  Dataset& data;
  /*! Feature mask */
  Mask& featureMask;
  /*! Index mask */
  Mask& indexMask;

 public:

  /*!
   * Parameter constructor.
   * \param _machine Machine to train.
   * \param _data Dataset to use for learning.
   * \param _featureMask Feature mask.
   * \param _indexMask Sample index mask.
   */
  Trainer(Machine& _machine, Dataset& _data, Mask& _featureMask, Mask& _indexMask);

  /*! 
   * Get the training dataset.
   * \return The training dataset.
   */
  Dataset& getTrainDataset() const;

  /*! 
   * Get the machine.
   * \return The learning machine.
   */
  Machine& getMachine() const;

  /*!
   * Get the feature mask.
   * \return The feature mask.
   */
  Mask& getFeatureMask() const;

  /*! 
   * Get the index mask.
   * \return The index map.
   */
  Mask& getIndexMask() const;

  /*!
   * Train the machine.
   */
  virtual void train() =0;
  
  /*!
   * Destructor.
   */
  ~Trainer();

};


#endif
