#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>

#include "../HnT.hpp"


using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  ClassificationDataset dataset;
  dataset.setName("Xor");
  dataset.addClass("positive");
  dataset.addClass("negative");
  int numPos=0;
  int numNeg=0;
  RNG random;
  for(int i=0;i<1000;i++){
    Mat tempMat(2,1,CV_64FC1,1.0);
    //random.fill(tempMat,RNG::NORMAL,0.0,1.0);
    vector<realv> meanValue = vector<realv>(0.0,1);
    vector<realv> stdValue = vector<realv>(1.0,1);
    random.fill(tempMat,RNG::NORMAL,meanValue,stdValue);
    FeatureVector tempVec(tempMat);
    if(tempVec[0]*tempVec[1]>0){
      dataset.addSequence(tempVec, "positive");
      numPos++;
    }
    else{
      dataset.addSequence(tempVec, "negative");
      numNeg++;
    }
  }
  cout << "Positives : " << numPos <<endl;
  cout << "Negatives : " << numNeg <<endl;
  cout << dataset.getMean();
  cout << dataset.getStandardDeviation();
  dataset.save("../xml/xor.xml");
  return EXIT_SUCCESS;
}
