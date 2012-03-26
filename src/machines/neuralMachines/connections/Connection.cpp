/*!
 * \file Connection.cpp
 * Body of the Connection class.
 * \author Luc Mioulet
 */

#include "Connection.hpp"

using namespace std;
using namespace cv;

Connection::Connection(Layer* _from, Layer* _to, uint _seed) : from(_from), to(_to), weights(cv::Mat()){
  int rows=to->getNumUnits();
  int cols=from->getNumUnits()+1;
  #ifdef REAL_DOUBLE
  weights = cv::Mat(rows,cols,CV_64FC1,0.0);
  #else
  weights = cv::Mat(rows,cols,CV_32FC1,0.0);
  #endif
  from->setOutputConnection(this);
  to->setInputConnection(this);
  initializeWeights(_seed);
}

Connection::Connection(Layer* _from, Layer* _to, cv::Mat _weights) : from(_from), to(_to), weights(_weights){
  int rows=to->getNumUnits();
  int cols=from->getNumUnits()+1;
  assert(rows==weights.rows);
  assert(cols==weights.cols);
}

Connection::Connection(const Connection& _c) :  from(_c.getInputLayer()->clone()), to(_c.getOutputLayer()->clone()){
  weights = _c.getWeights().clone();
}

Connection* Connection::clone() const{
  return new Connection(*this);
}

Mat Connection::getWeights() const{
  return weights;
}

Layer* Connection::getInputLayer() const{
  return from;
}

Layer* Connection::getOutputLayer() const{
  return to;
}

void Connection::setWeights(Mat _weights){
  if(to->getNumUnits()==((uint)weights.rows) && from->getNumUnits()+1==((uint)weights.cols)){
    weights=_weights;
  }
  else{
    throw("Connection : Wrong weight matrix size");
  }
}

void Connection::setInputLayer(Layer* _input){
  if(from->getNumUnits()+1==((uint)weights.cols)){
    from=_input;
  }
  else{
    throw("Connection : Wrong input size");
  }
}

void Connection::setOutputLayer(Layer* _output){
  if(to->getNumUnits()==((uint)weights.rows)){
    to=_output;   
  }
  else{
    throw("Connection : Wrong output size");
  }
}

void Connection::initializeWeights(uint _seed, realv _mean, realv _stdev){
  RNG random(_seed);
  random.next();
  random.fill(weights,RNG::NORMAL,_mean,_stdev);
}

Mat Connection::getWeightsToNeuron(int _i){
  return weights.row(_i);
}

Mat Connection::getWeightsFromNeuron(int _i){
  return weights.col(_i);
}

void Connection::forward(){
    to->forward();
}

void Connection::backwardDeltas(bool _output){
  from->backwardDeltas(_output);
}

void Connection::backwardWeights(realv _learningRate){
  ErrorVector ev=to->getErrorVector();
  for(int i=0;i<weights.rows;i++){
    for(int j=0;j<weights.cols;j++){
      weights.at<realv>(i,j)=weights.at<realv>(i,j)+_learningRate*ev[i]*from->getOutputSignal()[j];
    }
  }
  from->backwardWeights(_learningRate);
}

Connection::~Connection(){

}

ostream& operator<<(ostream& os, const Connection& c){
  os << "Connection between " << c.getInputLayer()->getName() <<" and " << c.getOutputLayer()->getName() << endl;
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
