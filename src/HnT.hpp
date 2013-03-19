#ifndef HNT_HPP_
#define HNT_HPP_

#include "General.hpp"
#include "dataset/Dataset.hpp"
#include "dataset/Mask.hpp"
#include "dataset/ValueVector.hpp"
#include "dataset/ErrorVector.hpp"
#include "dataset/FeatureVector.hpp"
#include "dataset/supervised/ClassificationDataset.hpp"
#include "dataset/supervised/RegressionDataset.hpp"
#include "dataset/supervised/SupervisedDataset.hpp"
#include "dataset/unsupervised/UnsupervisedDataset.hpp"
#include "machines/Machine.hpp"
#include "machines/neuralMachines/NeuralMachine.hpp"
#include "machines/neuralMachines/NeuralNetwork.hpp"
#include "machines/neuralMachines/PBDNN.hpp"
#include "machines/neuralMachines/layers/Layer.hpp"
#include "machines/neuralMachines/layers/LayerTanh.hpp"
#include "machines/neuralMachines/layers/LayerSigmoid.hpp"
#include "machines/neuralMachines/layers/LayerSoftMax.hpp"
#include "machines/neuralMachines/layers/InputLayer.hpp"
#include "machines/neuralMachines/layers/LayerCTC.hpp"
#include "machines/neuralMachines/connections/Connection.hpp"
#include "machines/neuralMachines/connections/FullConnection.hpp"
#include "machines/neuralMachines/connections/RecurrentConnection.hpp"
#include "trainer/Trainer.hpp"
#include "trainer/errorMeasurers/ErrorMeasurer.hpp"
#include "trainer/errorMeasurers/SEMeasurer.hpp"
#include "trainer/errorMeasurers/AEMeasurer.hpp"
#include "trainer/errorMeasurers/ClassificationErrorMeasurer.hpp"
#include "trainer/criteria/Criteria.hpp"
#include "trainer/criteria/Criterion.hpp"
#include "trainer/criteria/IterationCount.hpp"
#include "trainer/supervisedTrainers/SupervisedTrainer.hpp"
#include "performanceMeasurers/CrossValidationParams.hpp"
#include "performanceMeasurers/PerformanceMeasurer.hpp"
#include "performanceMeasurers/DiversityMeasurer.hpp"
#include "performanceMeasurers/RegressionMeasurer.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/NeuralNetworkTrainer.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/BackPropagation.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/CTCTrainer.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBP.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBPBatch.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/PopulationClusterBP.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/PopulationTrainer.hpp"
#include "trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.hpp"
#include "utilities/ImageProcessing.hpp"
#include "utilities/GeneralUtilities.hpp"
#include "utilities/TextUtilities.hpp"
#include "utilities/ToyDatasets.hpp"


#endif /* HNT_HPP_ */
