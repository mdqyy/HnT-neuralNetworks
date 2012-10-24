cmake_minimum_required(VERSION 2.8)
project(PhD)
#make flags
set(CMAKE_CXX_FLAGS "-Wall -pedantic")

find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS thread date_time program_options filesystem system REQUIRED)

find_package(Threads REQUIRED STATIC)

# check if doxygen is even installed
find_package(Doxygen)
if (DOXYGEN_FOUND STREQUAL "NO")
    message(FATAL_ERROR "Doxygen not found. Please get a copy http://www.doxygen.org")
endif (DOXYGEN_FOUND STREQUAL "NO")
if (DOXYGEN_FOUND STREQUAL "YES")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  add_custom_target(doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES doc)
  get_target_property(DOC_TARGET doc TYPE)
  if(NOT DOC_TARGET)
    add_custom_target(doc)
  endif()
  add_dependencies(doc doxygen)
endif (DOXYGEN_FOUND STREQUAL "YES")

add_subdirectory(src)
	
include_directories (/usr/include)
include_directories (/usr/local/include)
include_directories (src)

add_executable(executables/tests src/executables/mainTesting.cpp)
target_link_libraries(executables/tests Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createXORDB src/executables/createXORDB.cpp)
target_link_libraries(executables/createXORDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/neuralNet src/executables/nnTesting.cpp)
target_link_libraries(executables/neuralNet Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createPNDB src/executables/createPNDB.cpp)
target_link_libraries(executables/createPNDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/pbdnnTesting src/executables/pbdnnTesting.cpp)
target_link_libraries(executables/pbdnnTesting Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createPixelDB src/executables/createPixelDB.cpp)
target_link_libraries(executables/createPixelDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createRegressionPixelRimes src/executables/createRegressionPixelRimes.cpp)
target_link_libraries(executables/createRegressionPixelRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/createClassificationPixelRimes src/executables/createClassificationPixelRimes.cpp)
target_link_libraries(executables/createClassificationPixelRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBS})

add_executable(executables/pbdnnIAM src/executables/pbdnnIAM.cpp)
target_link_libraries(executables/pbdnnIAM Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnBatch src/executables/pbdnnBatch.cpp)
target_link_libraries(executables/pbdnnBatch Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnCluster src/executables/pbdnnCluster.cpp)
target_link_libraries(executables/pbdnnCluster Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterReload src/executables/pbdnnClusterReload.cpp)
target_link_libraries(executables/pbdnnClusterReload Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterIntermediateResults src/executables/pbdnnClusterIntermediateResults.cpp)
target_link_libraries(executables/pbdnnClusterIntermediateResults Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/pbdnnClusterDatCreator src/executables/pbdnnClusterDatCreator.cpp)
target_link_libraries(executables/pbdnnClusterDatCreator Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createImagesFromDB src/executables/createImagesFromDB.cpp)
target_link_libraries(executables/createImagesFromDB Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createPixelRimes src/executables/createPixelRimes.cpp)
target_link_libraries(executables/createPixelRimes Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/createClassificationRimesNetworks src/executables/createClassificationRimesNetworks.cpp)
target_link_libraries(executables/createClassificationRimesNetworks Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/testRNN src/executables/recurrentLearning.cpp)
target_link_libraries(executables/testRNN Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/diversityMeasurement src/executables/diversityMeasurement.cpp)
target_link_libraries(executables/diversityMeasurement Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})

add_executable(executables/CTCTesting src/executables/CTCTesting.cpp)
target_link_libraries(executables/CTCTesting Machines TinyXML ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Threads_LIBS})