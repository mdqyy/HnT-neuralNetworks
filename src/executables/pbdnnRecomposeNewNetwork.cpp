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
    arguments.push_back("txt file pointing to the different population files");
    arguments.push_back("validation set to find active networks");
    arguments.push_back("simple load");
    arguments.push_back("new population file");
    cout << helper("Create PBDNN by recomposing from different nets", "Create PBDNN using the best networks from a serie of trainings", arguments);
    if (argc != arguments.size() + 1) {
        cerr << "Not enough arguments" << endl;
        return EXIT_FAILURE;
    }
    ifstream populationFiles(argv[1]);
    bool sLoad = (atoi(argv[3])==1);
    RegressionDataset dataset;
    if(sLoad){
      dataset.simpleLoad(argv[2]);
    }
    else{
      dataset.load(argv[2]);
    }
    cout << "dataset loaded " << endl;
    vector<NeuralNetworkPtr> newPopulation = vector<NeuralNetworkPtr>();
    string line = "";
    AEMeasurer mae;
    vector<PBDNN> pops = vector<PBDNN>();
    /* explore all files */
    if(populationFiles.is_open()){
      while(populationFiles.good()){
        getline(populationFiles,line);
        ifstream popFile(line.c_str());
        if(line.size()>0){
          PBDNN pop;
          pops.push_back(pop);
          popFile >> pop;
          vector<NeuralNetworkPtr> nets =pop.getPopulation();
          DiversityMeasurer diversity(pop,dataset,mae);
          vector<int > assignedTo = diversity.sampleRepartition();
          for(uint i=0;i<assignedTo.size();i++){
        cout << "network " << i <<" from " << line << " got " << assignedTo[i]<< " elements " << endl;
        if(assignedTo[i]>0){
          cout << "added network" << i << endl;
          newPopulation.push_back(nets[i]);
        }
          }
        }
      }
    }
    PBDNN recomposedPopulation(newPopulation);
    ofstream outStream(argv[4]);
    outStream << recomposedPopulation;
    cout << "pop final size : " << newPopulation.size() << endl;
    return EXIT_SUCCESS;
}
