add_subdirectory(executables)
add_subdirectory(machines)
add_subdirectory(dataset)
add_subdirectory(performanceMeasurers)
add_subdirectory(trainer)
add_subdirectory(utilities)

include_directories (dataset)
include_directories (machines)
include_directories (tinyxml)
include_directories (performanceMeasurers)
include_directories (trainer)
include_directories (utilities)

add_library(Machines dataset/Dataset.cpp 
  dataset/Mask.cpp 
  dataset/ValueVector.cpp 
  dataset/ErrorVector.cpp 
  dataset/FeatureVector.cpp 
  dataset/supervised/ClassificationDataset.cpp
  dataset/supervised/RegressionDataset.cpp
  dataset/supervised/SupervisedDataset.cpp
  dataset/unsupervised/UnsupervisedDataset.cpp
  machines/Machine.cpp
  machines/neuralMachines/NeuralMachine.cpp
  machines/neuralMachines/NeuralNetwork.cpp
  machines/neuralMachines/PBDNN.cpp
  machines/neuralMachines/layers/Layer.cpp
  machines/neuralMachines/layers/LayerTanh.cpp
  machines/neuralMachines/layers/LayerSigmoid.cpp
  machines/neuralMachines/layers/LayerSoftMax.cpp
  machines/neuralMachines/layers/InputLayer.cpp
  machines/neuralMachines/layers/LayerCTC.cpp
  machines/neuralMachines/connections/Connection.cpp
  machines/neuralMachines/connections/FullConnection.cpp
  machines/neuralMachines/connections/RecurrentConnection.cpp
  trainer/Trainer.cpp
  trainer/errorMeasurers/ErrorMeasurer.cpp
  trainer/errorMeasurers/SEMeasurer.cpp
  trainer/errorMeasurers/AEMeasurer.cpp
  trainer/errorMeasurers/ClassificationErrorMeasurer.cpp
  trainer/criteria/Criteria.cpp
  trainer/criteria/Criterion.cpp
  trainer/criteria/IterationCount.cpp
  trainer/supervisedTrainers/SupervisedTrainer.cpp
  performanceMeasurers/CrossValidationParams.cpp
  performanceMeasurers/PerformanceMeasurer.cpp
  performanceMeasurers/DiversityMeasurer.cpp
  performanceMeasurers/RegressionMeasurer.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/NeuralNetworkTrainer.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/BackPropagation.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/CTCTrainer.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBP.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/PopulationBPBatch.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/PopulationClusterBP.cpp
  trainer/supervisedTrainers/neuralNetworkTrainers/LearningParams.cpp
  utilities/ImageProcessing.cpp
  utilities/TextUtilities.cpp
  utilities/ToyDatasets.cpp
)

add_library(TinyXML   
  tinyxml/tinyxml.cpp
  tinyxml/tinyxmlerror.cpp
  tinyxml/tinyxmlparser.cpp
  tinyxml/tinystr.cpp
)