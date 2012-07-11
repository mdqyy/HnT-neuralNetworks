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

#include "../dataset/supervised/ClassificationDataset.hpp"
#include "../dataset/ValueVector.hpp"
#include "../dataset/FeatureVector.hpp"
#include "../utilities/ImageProcessing.hpp"
#include "../utilities/TextUtilities.hpp"

using namespace std;
using namespace cv;

/*void fillImageVector(char* _directoryName,int _mode, RegressionDataset& _dataset, int _sectionLength){
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
}*/
/*
void load(std::string fileName, ClassificationDataset& dataset,){
	TiXmlDocument doc( fileName );
}*/

void addDictionaryClasses(ClassificationDataset* dataset){
	dataset->addClass("a");
	dataset->addClass("â");
	dataset->addClass("à");
	dataset->addClass("b");
	dataset->addClass("c");
	dataset->addClass("d");
	dataset->addClass("e");
	dataset->addClass("ê");
	dataset->addClass("é");
	dataset->addClass("è");
	dataset->addClass("f");
	dataset->addClass("g");
	dataset->addClass("h");
	dataset->addClass("i");
	dataset->addClass("j");
	dataset->addClass("k");
	dataset->addClass("l");
	dataset->addClass("m");
	dataset->addClass("n");
	dataset->addClass("o");
	dataset->addClass("p");
	dataset->addClass("q");
	dataset->addClass("r");
	dataset->addClass("s");
	dataset->addClass("t");
	dataset->addClass("u");
	dataset->addClass("ù");
	dataset->addClass("ü");
	dataset->addClass("v");
	dataset->addClass("w");
	dataset->addClass("x");
	dataset->addClass("y");
	dataset->addClass("z");
	dataset->addClass("A");
	dataset->addClass("B");
	dataset->addClass("C");
	dataset->addClass("D");
	dataset->addClass("E");
	dataset->addClass("F");
	dataset->addClass("G");
	dataset->addClass("H");
	dataset->addClass("I");
	dataset->addClass("J");
	dataset->addClass("K");
	dataset->addClass("L");
	dataset->addClass("M");
	dataset->addClass("N");
	dataset->addClass("O");
	dataset->addClass("P");
	dataset->addClass("Q");
	dataset->addClass("R");
	dataset->addClass("S");
	dataset->addClass("T");
	dataset->addClass("U");
	dataset->addClass("V");
	dataset->addClass("W");
	dataset->addClass("X");
	dataset->addClass("Y");
	dataset->addClass("Z");
	dataset->addClass("'");
	dataset->addClass("-");
	dataset->addClass("/");
	dataset->addClass("0");
	dataset->addClass("1");
	dataset->addClass("2");
	dataset->addClass("3");
	dataset->addClass("4");
	dataset->addClass("5");
	dataset->addClass("6");
	dataset->addClass("7");
	dataset->addClass("8");
	dataset->addClass("9");
}

void rimesLoader(string groundTruthFile, string groundTruthFolder,int frameSize, ClassificationDataset* dataset){
	string line, imageFile, label;
	size_t position;
	ifstream gtFile (groundTruthFile.c_str());
	Mat image;
	if (gtFile.is_open()){
		while (/*gtFile.good()*/ dataset->getNumSequences()<1){
			getline (gtFile,line);
			position = line.find(" ");
			label = line.substr(position+1);
			imageFile = groundTruthFolder + line.substr(0,position);
			image = imread(imageFile,0);
			if(!image.empty()){
				vector<FeatureVector> frames = extractFrames(image,frameSize);
				vector<string> labelSequence = extractLabelSequence(label);
				dataset->addSequence(frames,labelSequence);
			}
		}
	    gtFile.close();
	  }
}

int main (int argc, char* argv[]){
  ClassificationDataset dataset;
  addDictionaryClasses(&dataset);
  string groundTruthFile = argv[1];
  string groundTruthFolder = argv[2];
  int frameSize = atoi(argv[3]);
  dataset.setName(argv[4]);
  string saveLocation = argv[5];
  rimesLoader(groundTruthFile, groundTruthFolder, frameSize, &dataset);
  dataset.save(saveLocation);
  namedWindow("Deslanted image", CV_WINDOW_NORMAL);
  return EXIT_SUCCESS;
}
