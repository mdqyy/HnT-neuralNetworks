#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <stdio.h>
#include <fstream>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

void fillImageVector(char* _directoryName,int _mode, RegressionDataset& _dataset, int _sectionLength){
  DIR *dp;
  struct dirent *ep;
  dp = opendir (_directoryName);
  if (dp != NULL){
    while (ep = readdir (dp)){
      //append an image to the vector only if it has a .png extension
      if(strstr(ep->d_name,".png")!=NULL){
	char str[200]="";
	strcat(str,_directoryName);
	strcat(str,"/");
	strcat(str,ep->d_name);
	Mat image = imread(str,_mode);
	int subparts=floor(((float)image.cols)/((float)_sectionLength));
	vector<FeatureVector> features;
	for(int i=0;i<subparts;i++){
	  FeatureVector fv(_sectionLength*image.rows);
	  for(int j=0;j<_sectionLength;j++){
	    for(int k=0;k<image.rows;k++){
	      if((int)image.at<uchar>(k,i*_sectionLength+j)==255){
		fv[j*image.rows+k]=1.0;
	      }
	      else{
		fv[j*image.rows+k]=0.0;
	      }
	    }
	  }
	  features.push_back(fv);
	}
	_dataset.addSequence(features,features);
      }
    }
    (void) closedir (dp);
  }
  else{
    cerr << "ERROR in CreateImageVector : Couldn't open the directory";
    exit(1);
  }
}

int main (int argc, char* argv[]){
  RegressionDataset dataset;
  dataset.setName("IAM-sequenced10px");
  
  fillImageVector(argv[1], 0, dataset, 10);
  cout << dataset.getMean();
  cout << dataset.getStandardDeviation();
  dataset.save("../xml/IAM-10.xml");
  return EXIT_SUCCESS;
}
