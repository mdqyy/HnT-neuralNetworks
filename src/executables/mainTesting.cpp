#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../machines/neuralMachines/layers/InputLayer.hpp"
#include "../machines/neuralMachines/layers/LayerTanh.hpp"
#include "../machines/neuralMachines/connections/Connection.hpp"

using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  ClassificationDataset dataset("test.xml");
  Mat testMat(5,1,CV_64FC1,1.0);
  Mat meanMat(5,1,CV_64FC1,0.0);
  Mat stdevMat(5,1,CV_64FC1,1.0);
  FeatureVector testFv(testMat);
  ValueVector mean(meanMat);
  ValueVector stdev(stdevMat);
  InputLayer il(5,mean,stdevMat);
  cout << testFv ;
  LayerTanh th(2);
  cout << th ;
  Connection c(il,th);
  cout << th ;
  cout << il;
  cout << c;
  il.forward(testFv);
  cout << th.getOutputSignal();
  cout << dataset;
  dataset.save("temp.xml");
  return EXIT_SUCCESS;
}
