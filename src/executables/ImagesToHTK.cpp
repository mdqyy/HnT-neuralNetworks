#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <iostream>
#define _POSIX_SOURCE
#include <sys/stat.h>
#undef _POSIX_SOURCE

#include <boost/thread/thread.hpp>
#include <sstream>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <map>

#include "../HnT.hpp"

using namespace std;
using namespace cv;


// Thread process per network only forward to find min error networks
void threadForwardPerNetworkError(vector<NeuralNetworkPtr>* _population, uint _i, vector<FeatureVector>* _error, vector<FeatureVector>* _frames){
  AEMeasurer mae;
  for (uint k = 0; k < (*_frames).size(); k++) {
    FeatureVector sample = (*_frames)[k];
    FeatureVector networkOutput;
    (*_population)[_i]->forward(sample);
    networkOutput = (*_population)[_i]->getOutputSignal();
    (*_error)[k][_i] = mae.totalError(networkOutput, sample);
  }
}




int main(int argc, char* argv[]) {
  vector<string> arguments;
  arguments.push_back("image file");
  arguments.push_back("folder containing the images");
  arguments.push_back("imageHeight");
  arguments.push_back("frame size");
  arguments.push_back("save folder for the dataset");
  arguments.push_back("model length, to suppres short training samples");
  arguments.push_back("population file");
  arguments.push_back("train (0) or validation/test (1)");
  arguments.push_back("dictionary size if test mode");
  cout << helper("Create HTK learning files", "Create HTK learning files using images and a network.", arguments) << endl;

  if (argc != arguments.size() + 1) {
    cerr << "Not enough arguments, " << argc-1 << " given and "<< arguments.size()<<" required" << endl;
    return EXIT_FAILURE;
  }
  string groundTruthFile = argv[1];
  string groundTruthFolder = argv[2];
  int imageHeight = atoi(argv[3]);
  int frameSize = atoi(argv[4]);
  string saveLocation = argv[5]; 
  int modelLength = atoi(argv[6]);
  PBDNN pop;
  ifstream inStream(argv[7]);
  bool testMode = (1==atoi(argv[8]));
  uint dictionnarySize = atoi(argv[9]);
  inStream >> pop;
  vector<string> words = vector<string>();
  vector<NeuralNetworkPtr> population = pop.getPopulation();
  ostringstream dictionnaryFileLocation;
  ostringstream htkScpFileLocation;
  ostringstream labScpFileLocation;
  map<string,string> dictionnary;
  dictionnaryFileLocation << saveLocation << "dictionnary.scp";
  htkScpFileLocation << saveLocation << "dataFiles.scp";
  labScpFileLocation << saveLocation << "labFiles.scp";
  ofstream dictionnaryFile(dictionnaryFileLocation.str().c_str());
  ofstream htkScp(htkScpFileLocation.str().c_str());
  ofstream labScp(labScpFileLocation.str().c_str());  

  ostringstream htkScpFolder;
  ostringstream labTrainFolder;
  ostringstream labInitFolder;
  ostringstream dictionnariesFolderTM;
  ostringstream labFilesFolderTM;
  ostringstream htkFilesFolderTM;
  htkScpFolder << saveLocation << "htkn/";
  labInitFolder << saveLocation << "lab/";
  labTrainFolder << saveLocation << "labTrain/";
  dictionnariesFolderTM << saveLocation << "dictionnaries/";
  labFilesFolderTM << saveLocation << "labFiles/";
  htkFilesFolderTM << saveLocation << "dataFiles/";
  if (mkdir(htkScpFolder.str().c_str(), S_IRWXU) == 0){
    cout << "htk folder created" << endl;
  }
  if (mkdir(labTrainFolder.str().c_str(), S_IRWXU) == 0){
    cout << "lab train folder created" << endl;
  }
  if (mkdir(labInitFolder.str().c_str(), S_IRWXU) == 0){
    cout << "lab init folder created" << endl;
  }
  if (testMode){
      if (mkdir(dictionnariesFolderTM.str().c_str(), S_IRWXU) == 0){
	cout << "dictionnaries folder created" << endl;
      }
      if (mkdir(labFilesFolderTM.str().c_str(), S_IRWXU) == 0){
	cout << "lab files folder created" << endl;
      }
      if (mkdir(htkFilesFolderTM.str().c_str(), S_IRWXU) == 0){
	cout << "htk files folder created" << endl;
      }
  }

  string line, imageFile, label, name;
  size_t position;
  ifstream gtFile(groundTruthFile.c_str());
  Mat image;
  AEMeasurer ae;
  int j = 0;
  if (gtFile.is_open()) {
    while (gtFile.good()) {
      getline(gtFile, line);
      position = line.find(" ");
      label = line.substr(position + 1);
      imageFile = groundTruthFolder + line.substr(0, position);
      label = rtrim(label);
      image = imread(imageFile, 0);
      if (!image.empty() && image.rows == imageHeight) {
	vector<FeatureVector> frames = extractOverlappingFramesPPerP(image, frameSize);
	vector<string> labelSequence = extractLabelSequence(label);
	words.push_back(label);
	if(frames.size()>labelSequence.size()*modelLength){
	  ostringstream sequenceFile, wordFile,wordFile2;
	  ostringstream word, spacedLetters;
	  sequenceFile << saveLocation << "htkn/" << j <<".htk";
	  wordFile << saveLocation << "lab/"<<j <<".lab";
	  wordFile2 << saveLocation << "labTrain/"<<j <<".lab";
	  htkScp << sequenceFile.str() << endl;
	  labScp << wordFile.str() << endl;
	  
	  ofstream outputSequence(sequenceFile.str().c_str());
	  ofstream outputWord(wordFile.str().c_str());
	  ofstream outputWord2(wordFile2.str().c_str());

	  vector<FeatureVector> errors = vector<FeatureVector>(frames.size(), FeatureVector(population.size()) );
	  vector<boost::thread * > threadsForward;
	  for(uint k=0; k<population.size();k++){
	    threadsForward.push_back(new boost::thread(threadForwardPerNetworkError,&population, k, &errors, &frames));
	  }
	  for(uint k=0; k<population.size();k++){
	    threadsForward[k]->join();
	    delete threadsForward[k];
	  }
	  for (uint k = 0; k < frames.size(); k++) {
	    for (uint i = 0; i < population.size(); i++) {
	    	      outputSequence << errors[k][i] <<" " ;
	    }
	    outputSequence << endl;
	  }

	  for(uint k=0; k < labelSequence.size();k++){
	    outputWord << labelSequence[k] << endl;
	    outputWord2 << labelSequence[k] ;
	    word << labelSequence[k] ;
	    spacedLetters << labelSequence[k] << " ";
	  }
	  dictionnary.insert(pair<string,string>(word.str(),spacedLetters.str()));
	  outputWord.close();
	  outputSequence.close();
	}
	else{
	  cout << frames.size() << " " << labelSequence.size() << endl;
	  cout << imageFile <<" "<<label<< " too short to be added" << endl;
	}
      }
      else {
	cout << "Is empty label : " << label << endl;
	cout << "In file : "<< imageFile << endl;
      }
      j++;
    }
    gtFile.close();
  }
  
  map<string,string>::iterator it ;
  for(it=dictionnary.begin();it!=dictionnary.end();it++){
    dictionnaryFile << (*it).first << " " << (*it).second << endl;
  }
  if(testMode && dictionnary.size()<dictionnarySize){
    cout <<" dictionary is too small, "<< dictionnary.size() <<" unique words and " << dictionnarySize << " required." << endl;
    return EXIT_FAILURE;
  }

  /*TADAAAAA */
  if(testMode){
    for(uint j=0;j<words.size();j++){
      ostringstream dictionnaryFileLocation, sequenceFile, wordFile, htkScpFileLocation, labScpFileLocation;
      dictionnaryFileLocation << saveLocation << "dictionnaries/" << j << ".scp";
      htkScpFileLocation << saveLocation << "dataFiles/" << j << ".scp";
      labScpFileLocation << saveLocation << "labFiles/" << j << ".scp";
      sequenceFile << saveLocation << "htk/" << j << ".htk";
      wordFile << saveLocation << "lab/" << j << ".lab";
      ofstream dictionnaryFileTM(dictionnaryFileLocation.str().c_str());
      ofstream htkScpTM(htkScpFileLocation.str().c_str());
      ofstream labScpTM(labScpFileLocation.str().c_str());
      ostringstream word, spacedLetters;
      map<string, string> dictionnaryTM;
      dictionnaryTM.insert(pair<string, string>(words[j], dictionnary[words[j]]));
      RNG randomK((uint) getTickCount());
      while (dictionnaryTM.size() < dictionnarySize) {
	ostringstream word2, spacedLetters2;
	uint randK = 0;
	do {
	  randK = randomK.uniform(0, words.size());
	} while (randK == j);
	dictionnaryTM.insert(pair<string, string>(words[randK], dictionnary[words[randK]]));
      }
      map<string, string>::iterator it;
      for (it = dictionnaryTM.begin(); it != dictionnaryTM.end(); it++) {
	dictionnaryFileTM << (*it).first << " " << (*it).second << endl;
      }
      htkScpTM << sequenceFile.str() << " " << endl ;
      labScpTM << wordFile.str() << " " << endl;
      
    }
  }
  return EXIT_SUCCESS;
}
