#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <list>
#include <vector>
#define _POSIX_SOURCE
#include <sys/stat.h>
#include <unistd.h>
#undef _POSIX_SOURCE
#include <stdio.h>

#include <sstream>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  vector<string> arguments;
  arguments.push_back("mixed ensemble population");
  arguments.push_back("training image dataset");
  arguments.push_back("space between each frame");
  cout << helper("Mixed ensemble training", "Only trains the last layer, not the complete architecture", arguments) << endl;
  if (argc -1 != arguments.size()) {
    cerr << "Not enough arguments, " << argc - 1 << " given and " << arguments.size() <<" required" << endl;
    return EXIT_FAILURE;
  }
  ifstream is(argv[1]);
  string trainingFile(argv[2]);
  uint interFrameSpace = atoi(argv[3]);
  MixedEnsembles me;
  is >> me ;
  ImageDataset dataset;
  dataset.load(trainingFile);
  vector<ImageFrameExtractor> ifes = me.getIFEs();
  vector<string> windows = vector<string>();
  bool isActivated = true;
  int imageIndex = 0;
  int colIndex = 0;
  namedWindow("original");
  for(uint i= 0;i < ifes.size();i++){
      ostringstream name;
      name << "ife" << i;
      namedWindow(name.str(),CV_WINDOW_NORMAL);
      windows.push_back(name.str());
  }
  while(isActivated){
    Mat originalImage = dataset.getMatrix(imageIndex,0);
    for(uint i= 0;i < ifes.size();i++){
      FeatureVector fv = ifes[i].getFrameCenteredOn(originalImage,colIndex);
      Mat imageIFE = buildFrame(fv,ifes[i].getFrameSize());
      imshow(windows[i],imageIFE);
    }
    Mat drawnImage =  dataset.getMatrix(imageIndex,0);
    line(drawnImage,Point(colIndex,0),Point(colIndex,drawnImage.rows),Scalar(255),1);
    imshow("original",drawnImage);
    uchar key = waitKey(0);
    if('x' == key ){
      isActivated =false;
    }
    if('i' == key){
        if(colIndex + interFrameSpace < originalImage.cols){
          colIndex += interFrameSpace;
        }
    }
    if('a' == key){
        if(colIndex - interFrameSpace >= 0){
          colIndex -= interFrameSpace;
        }
    }
     if('m' == key){
        if(imageIndex +1 < dataset.getNumberOfImages()){
          imageIndex ++;
          colIndex = 0;
        }
     }
      if('r' == key){
        if(imageIndex -1 >= 0){
          imageIndex --;
          colIndex = 0;
        }
      }


  }
  return EXIT_SUCCESS;
}
