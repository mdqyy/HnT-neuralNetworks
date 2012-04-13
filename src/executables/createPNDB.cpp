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
  ClassificationDataset dataset;
  dataset.setName("PN");
  dataset.addClass("positive");
  dataset.addClass("negative");
  RNG random;
  for(int i=0;i<100;i++){
    Mat tempMat(1,1,CV_64FC1,1.0);
    random.next();
    random.fill(tempMat,RNG::NORMAL,0.0,1.0);
    FeatureVector tempVec(tempMat);
    if(tempVec[0]>0){
      dataset.addSequence(tempVec, "positive");
    }
    else{
      dataset.addSequence(tempVec, "negative");
    }
  }
  //  cout << dataset;
  cout << dataset.getMean();
  cout << dataset.getStandardDeviation();
  dataset.save("../xml/pn.xml");
  return EXIT_SUCCESS;
}
