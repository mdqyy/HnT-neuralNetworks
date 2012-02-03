/*!
 * \file Connection.cpp
 * Body of the Connection class.
 * \author Luc Mioulet
 */

#include "Connection.hpp"

using namespace std;
using namespace cv;

Connection::Connection(Layer& _from, Layer& _to, uint _seed) : from(_from), to(_to), weights(cv::Mat()){
  int rows=to.getNumUnits();
  int cols=from.getNumUnits();
  #ifdef REAL_DOUBLE
  weights = cv::Mat(rows,cols,CV_64FC1,0.0);
  #else
  weights = cv::Mat(rows,cols,CV_32FC1,0.0);
  #endif
  from.addOutputConnections(this);
  to.addInputConnections(this);
  initializeWeights(_seed);
}

Connection::Connection(Layer& _from, Layer& _to, cv::Mat _weights) : from(_from), to(_to), weights(_weights){
  int rows=to.getNumUnits();
  int cols=from.getNumUnits();
  assert(rows==weights.rows);
  assert(cols==weights.cols);
}

Mat Connection::getWeights() const{
  return weights;
}

Layer& Connection::getInputLayer() const{
  return from;
}

Layer& Connection::getOutputLayer() const{
  return to;
}

void Connection::setWeights(Mat _weights){
  if(to.getNumUnits()==((uint)weights.rows) && from.getNumUnits()==((uint)weights.cols)){
    weights=_weights;
  }
  else{
    throw("Wrong weight matrix size");
  }
}

void Connection::setInputLayer(Layer& _input){
  if(from.getNumUnits()==((uint)weights.cols)){
    from=_input;
  }
  else{
    throw("Wrong input size");
  }
}

void Connection::setOutputLayer(Layer& _output){
  if(to.getNumUnits()==((uint)weights.rows)){
    to=_output;   
  }
  else{
    throw("Wrong output size");
  }
}

void Connection::initializeWeights(uint seed, realv mean, realv stdev){
  RNG random(seed);
  random.fill(weights,RNG::NORMAL,mean,stdev);
}

Mat Connection::getWeightsNeuron(int i){
  return weights.row(i);
}

void Connection::forward(){
  to.forward();
}

Connection::~Connection(){

}

ostream& operator<<(ostream& os, const Connection& c){
  os << "Connection between " << c.getInputLayer().getName() <<" and " << c.getOutputLayer().getName() << endl;
  os << " Weight matrix " << endl;
  Mat temp = c.getWeights();
  for(int i=0;i< temp.rows;i++){
    for(int j=0;j< temp.cols;j++){
      os << temp.at<realv>(i,j) <<" ; " ;
    }
    os << endl;
  }
  return os;
}
