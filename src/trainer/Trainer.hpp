#ifndef __TRAINER_HPP__
#define __TRAINER_HPP__
/*!
 * \file Trainer.hpp
 * Header of the Trainer class.
 * \author Luc Mioulet
 */

#include "../dataset/Dataset.hpp"
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

 public:

  /*!
   * Parameter constructor.
   * \param _machine Machine to train.
   * \param _data Dataset to use for learning.
   */
  Trainer(Machine& _machine, Dataset& _data);

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
   * Train the machine.
   */
  virtual void train() =0;
  
  /*!
   * Destructor.
   */
  ~Trainer();

};


#endif
