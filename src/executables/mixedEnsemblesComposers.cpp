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

void suppressLastLayers(vector<NeuralNetworkPtr> _nets) {
  for (uint i = 0; i < _nets.size(); i++) {
    _nets[i]->suppressLastLayer();
  }
}

int main(int argc, char* argv[]) {
  vector<string> arguments;
  arguments.push_back("output net hidden neurons");
  arguments.push_back("output net output neurons");
  arguments.push_back("population file");
  arguments.push_back("ife scale");
  arguments.push_back("ife frame size");
  arguments.push_back("ife inter frame size");
  cout << helper("Mixed ensemble connector", "Connector test", arguments) << endl;
  if ((argc - 3) % 4 != 0) {
    cerr << "Not enough arguments, " << argc - 1 << " given and 2 + 4 times X required" << endl;
    return EXIT_FAILURE;
  }
  uint hiddenNeurons = atoi(argv[1]);
  uint outputNeurons = atoi(argv[2]);
  uint inputs = 0;
  uint numberOfPops = (argc - 3) / ((int) arguments.size() - 2);
  cout << "num pops " << numberOfPops << " arg size " << arguments.size() << endl;
  vector<NeuralNetworkPtr> nets = vector<NeuralNetworkPtr>();
  vector<ImageFrameExtractor> ifes = vector<ImageFrameExtractor>();
  vector<uint> links = vector<uint>();
  for (uint i = 0; i < numberOfPops; i++) {
    char* name = argv[i * arguments.size() + 3];
    realv scale = atof(argv[i * arguments.size() + 4]);
    uint frameSize = atoi(argv[i * arguments.size() + 5]);
    uint interFrameSpace = atoi(argv[i * arguments.size() + 6]);
    ifstream in(name);
    PBDNN pop;
    in >> pop;
    vector<NeuralNetworkPtr> localNets = pop.getPopulation();
    suppressLastLayers(localNets);
    ImageFrameExtractor ife(scale, frameSize, interFrameSpace);
    for (uint j = 0; j < localNets.size(); j++) {
      links.push_back(i);
      inputs += localNets[j]->getHiddenLayers().back()->getNumUnits();
    }
    nets.insert(nets.end(),localNets.begin(),localNets.end());
    ifes.push_back(ImageFrameExtractor(scale, frameSize, interFrameSpace));
  }
  cout << "inputs " << inputs << endl;
  RNG random(getTickCount());
  random.next();
  ValueVector mean(inputs); /*no idea how to initialize this, yabi(yet another bad idea)*/
  ValueVector stdDev(inputs); /* idem */
  LayerPtr il = LayerPtr(new InputLayer(inputs, mean, stdDev));
  LayerPtr th = LayerPtr(new LayerSigmoid(hiddenNeurons));
  LayerPtr out = LayerPtr(new LayerSigmoid(outputNeurons));
  ConnectionPtr c1 = ConnectionPtr(
      new Connection(il.get(), th.get(), random.next()));
  ConnectionPtr c2 = ConnectionPtr(
      new Connection(th.get(), out.get(), random.next()));
  Mat ts = c1->getWeights();
  Mat td = c2->getWeights();
  for (int i = 0; i < ts.cols - 1; i++) {
    for (int j = i; j < td.cols - 1; j++) {
      td.at<realv>(i, j) = ts.at<realv>(j, i);
    }
  }
  c2->setWeights(td.clone());
  vector<LayerPtr> layers;
  layers.push_back(il);
  layers.push_back(th);
  layers.push_back(out);
  vector<ConnectionPtr> connections;
  connections.push_back(c1);
  connections.push_back(c2);
  NeuralNetworkPtr net = NeuralNetworkPtr(
      new NeuralNetwork(layers, connections, "network"));
  MixedEnsembles me = MixedEnsembles(nets, ifes, links, net);
  ofstream os("mixedEnsemble.pop");
  os << me;
  return EXIT_SUCCESS;
}
