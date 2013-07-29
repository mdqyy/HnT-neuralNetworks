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
#include <sstream>cout << i << endl;

#include "../HnT.hpp"



using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    vector<string> arguments;
    arguments.push_back("population file");
    arguments.push_back("new population file");
    arguments.push_back("after that add all networks to keep");
    cout << helper("Create PBDNN by recomposing from different nets", "Create PBDNN using the best networks from a serie of trainings, you get what you want", arguments);
    if (argc < arguments.size() + 1) {
        cerr << "Not enough arguments" << endl;
        return EXIT_FAILURE;
    }
    ifstream populationFile(argv[1]);
    PBDNN pop;
    populationFile >> pop;

    vector<NeuralNetworkPtr> newPopulation = vector<NeuralNetworkPtr>();


    vector<NeuralNetworkPtr> nets =pop.getPopulation();

    for(uint i=3;i<argc;i++){
        cout << argv[i] << endl;
        uint index = atoi(argv[i]);
        cout << "added network" << index << endl;
        newPopulation.push_back(nets[index]);
    }

    PBDNN recomposedPopulation(newPopulation);
    ofstream outStream(argv[2]);
    outStream << recomposedPopulation;
    cout << "pop final size : " << newPopulation.size() << endl;
    return EXIT_SUCCESS;
}
