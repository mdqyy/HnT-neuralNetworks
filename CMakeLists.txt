cmake_minimum_required(VERSION 2.8)
project(PhD)
#make flags
set(CMAKE_CXX_FLAGS "-Wall -pedantic -std=c99 ")

#$SET(OpenCV_DIR "/home/entitees/labo-litis/partages/maurdor-labo-litis/libraries_install/share/OpenCV")
find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS thread date_time program_options filesystem system REQUIRED)

find_package(Threads REQUIRED STATIC)

# check if doxygen is even installed

add_subdirectory(src)
	
include_directories (/usr/include)
include_directories (/usr/local/include)
include_directories (src)

##Simple tests

add_executable(executables/tests src/executables/mainTesting.cpp)
target_link_libraries(executables/tests Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/neuralNet src/executables/nnTesting.cpp)
target_link_libraries(executables/neuralNet Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/pbdnnTesting src/executables/pbdnnTesting.cpp)
target_link_libraries(executables/pbdnnTesting Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/CTCTesting src/executables/CTCTesting.cpp)
target_link_libraries(executables/CTCTesting Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/testRNN src/executables/recurrentLearning.cpp)
target_link_libraries(executables/testRNN Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/connectorTest src/executables/connectorTest.cpp)
target_link_libraries(executables/connectorTest Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})


##Ensemble training

add_executable(executables/pbdnnIAM src/executables/pbdnnIAM.cpp)
target_link_libraries(executables/pbdnnIAM Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnBatch src/executables/pbdnnBatch.cpp)
target_link_libraries(executables/pbdnnBatch Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnCluster src/executables/pbdnnCluster.cpp)
target_link_libraries(executables/pbdnnCluster Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterForToys src/executables/pbdnnClusterForToys.cpp)
target_link_libraries(executables/pbdnnClusterForToys Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterReload src/executables/pbdnnClusterReload.cpp)
target_link_libraries(executables/pbdnnClusterReload Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterBestNetwork src/executables/pbdnnClusterBestNetwork.cpp)
target_link_libraries(executables/pbdnnClusterBestNetwork Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnTrainer src/executables/pbdnnTrainer.cpp)
target_link_libraries(executables/pbdnnTrainer Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnInverseTrainer src/executables/pbdnnInverseTrainer.cpp)
target_link_libraries(executables/pbdnnInverseTrainer Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnRecomposer src/executables/pbdnnRecomposeNewNetwork.cpp)
target_link_libraries(executables/pbdnnRecomposer Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnSuppressNets src/executables/pbdnnSuppressNets.cpp)
target_link_libraries(executables/pbdnnSuppressNets Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnTrainerPursuit src/executables/pbdnnTrainerPursuit.cpp)
target_link_libraries(executables/pbdnnTrainerPursuit Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnImageInverseTrainer src/executables/pbdnnImageInverseTrainer.cpp)
target_link_libraries(executables/pbdnnImageInverseTrainer Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})


##Create databases 

add_executable(executables/createXORDB src/executables/createXORDB.cpp)
target_link_libraries(executables/createXORDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createPNDB src/executables/createPNDB.cpp)
target_link_libraries(executables/createPNDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createImagesFromDB src/executables/createImagesFromDB.cpp)
target_link_libraries(executables/createImagesFromDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createImagesFromNetwork src/executables/createImageFromNetwork.cpp)
target_link_libraries(executables/createImagesFromNetwork Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createPixelRimes src/executables/createPixelRimes.cpp)
target_link_libraries(executables/createPixelRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createClassificationRimesNetworks src/executables/createClassificationRimesNetworks.cpp)
target_link_libraries(executables/createClassificationRimesNetworks Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createPixelDB src/executables/createPixelDB.cpp)
target_link_libraries(executables/createPixelDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createRegressionPixelRimes src/executables/createRegressionPixelRimes.cpp)
target_link_libraries(executables/createRegressionPixelRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createRegressionPixelRimesIFE src/executables/createRegressionPixelRimesIFE.cpp)
target_link_libraries(executables/createRegressionPixelRimesIFE Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createRegressionPixelRimesPerPart src/executables/createRegressionPixelRimesPerPart.cpp)
target_link_libraries(executables/createRegressionPixelRimesPerPart Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createClassificationPixelRimes src/executables/createClassificationPixelRimes.cpp)
target_link_libraries(executables/createClassificationPixelRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createHTKFiles src/executables/createHTKFiles.cpp)
target_link_libraries(executables/createHTKFiles Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createHTKHundredDictionnaries src/executables/createHTKHundredDictionnariesFiles.cpp)
target_link_libraries(executables/createHTKHundredDictionnaries Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createToyClusterDatasets src/executables/createToyClusterDatasets.cpp)
target_link_libraries(executables/createToyClusterDatasets Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createSignalFromNetwork src/executables/createSignalFromNetwork.cpp)
target_link_libraries(executables/createSignalFromNetwork Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createImageFromClusters src/executables/createImageFromClusters.cpp)
target_link_libraries(executables/createImageFromClusters Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/imagesToHTK src/executables/ImagesToHTK.cpp)
target_link_libraries(executables/imagesToHTK Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES}  ${Boost_LIBS})

add_executable(executables/createImageDataset src/executables/createImageDataset.cpp)
target_link_libraries(executables/createImageDataset Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createClassDataset src/executables/createClassDataset.cpp)
target_link_libraries(executables/createClassDataset Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})


## Various tasks

add_executable(executables/pbdnnClusterIntermediateResults src/executables/pbdnnClusterIntermediateResults.cpp)
target_link_libraries(executables/pbdnnClusterIntermediateResults Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterDatCreator src/executables/pbdnnClusterDatCreator.cpp)
target_link_libraries(executables/pbdnnClusterDatCreator Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/diversityMeasurement src/executables/diversityMeasurement.cpp)
target_link_libraries(executables/diversityMeasurement Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createFoldRimes src/executables/createFoldRimes.cpp)
target_link_libraries(executables/createFoldRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/meanImage src/executables/meanImage.cpp)
target_link_libraries(executables/meanImage Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/weightImages src/executables/weightImages.cpp)
target_link_libraries(executables/weightImages Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})


## Mixed ensembles

add_executable(executables/mixedEnsemblesComposers src/executables/mixedEnsemblesComposers.cpp)
target_link_libraries(executables/mixedEnsemblesComposers Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/mixedEnsemblesCTCAdder src/executables/mixedEnsemblesCTCAdder.cpp)
target_link_libraries(executables/mixedEnsemblesCTCAdder Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/mixedEnsemblesCTCTraining src/executables/mixedEnsemblesCTCTraining.cpp)
target_link_libraries(executables/mixedEnsemblesCTCTraining Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/mixedEnsemblesTraining src/executables/mixedEnsemblesTraining.cpp)
target_link_libraries(executables/mixedEnsemblesTraining Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/mixedEnsemblesVisualisation src/executables/mixedEnsemblesVisualisation.cpp)
target_link_libraries(executables/mixedEnsemblesVisualisation Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/mixedEnsemblesClassificationTraining src/executables/mixedEnsemblesClassificationTraining.cpp)
target_link_libraries(executables/mixedEnsemblesClassificationTraining Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})