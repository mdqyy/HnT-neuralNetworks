/*!
 * \file Connection.cpp
 * Body of the Connection class.
 * \author Luc Mioulet
 */

#include "Connection.hpp"

using namespace std;
using namespace cv;

Connection::Connection() : from(0),to(0),weights(cv::Mat()){
}

Connection::Connection(Layer* _from, Layer* _to, uint _seed) : from(_from), to(_to), weights(cv::Mat()){
  int rows=to->getNumUnits();
  int cols=0;
  if(to->isRecurrent()){
    cols = from->getNumUnits()+rows +1;
  }
  else{
    cols = from->getNumUnits()+1;
  }

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
  int cols=0;
  if(to->isRecurrent()){
    cols = from->getNumUnits()+rows +1;
  }
  else{
    cols = from->getNumUnits()+1;
  }
  assert(rows==weights.rows);
  assert(cols==weights.cols);
}

Connection::Connection(const Connection& _c) :  from(0), to(0){
  if(_c.getInputLayer()!=0){
    from = _c.getInputLayer()->clone();
  }
  if(_c.getOutputLayer()!=0){
    to = _c.getOutputLayer()->clone();
  }
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
  if(from!=0 && to!=0){
    if(to->getNumUnits()==((uint)weights.rows) && (from->getNumUnits()+1==((uint)weights.cols) || (from->getNumUnits()+1+to->getNumUnits()==((uint)weights.cols)&& to->isRecurrent())) ){
      weights=_weights;
    }
    else{
      throw("Connection : Wrong weight matrix size");
    }
  }
  else{
    weights=_weights;
  }
}

void Connection::setInputLayer(Layer* _input){
  if(_input->getNumUnits()+1>=((uint)weights.cols)){ /*! Should changethis shitty condition */
    from=_input;
  }
  else{
    throw("Connection : Wrong input size");
  }
}

void Connection::setOutputLayer(Layer* _output){
  if(_output->getNumUnits()==((uint)weights.rows)){
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
  if(from!=to){
    from->backwardDeltas(_output);
  }
}

void Connection::backwardWeights(realv _learningRate){
  ErrorVector ev=to->getErrorVector();
  for(int i=0;i<weights.rows;i++){
    for(int j=0;j<weights.cols;j++){
      weights.at<realv>(i,j)=weights.at<realv>(i,j)+_learningRate*ev[i]*from->getOutputSignal()[j];
    }
  }
  if(from!=to){
    from->backwardWeights(_learningRate);
  }
}

Connection::~Connection(){
  from = 0;
  to = 0;
}

ostream& operator<<(ostream& os, const Connection& c){
  os << "Connection";
  if(c.getInputLayer()!=0 && c.getOutputLayer()!=0){
    os <<" between "<<  c.getInputLayer()->getName() <<" and " << c.getOutputLayer()->getName();
  }
  os << endl;
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

ofstream& operator<<(ofstream& ofs, const Connection& l){
  Mat weightsTmp = l.getWeights();
  ofs << "< " << weightsTmp.rows << " "<< weightsTmp.cols <<" [" ;
  for(int i =0; i<weightsTmp.rows;i++){
    for(int j=0;j<weightsTmp.cols;j++){
      ofs << " "<< weightsTmp.at<realv>(i,j);
    }
    ofs << endl;
  }
  ofs <<"] >"<< endl;
  return ofs;
}

ifstream& operator>>(ifstream& ifs, Connection& l){
  Mat weights;
  string temp;
  int cols,rows;
  ifs >> temp;
  ifs >> rows;
  ifs >> cols;
  ifs >> temp;
  #ifdef REAL_DOUBLE
  weights = cv::Mat(rows,cols,CV_64FC1,0.0);
  #else
  weights = cv::Mat(rows,cols,CV_32FC1,0.0);
  #endif
  realv value;
  for(int i=0; i<rows;i++){
    for(int j=0;j<cols;j++){
      ifs >> value;
      weights.at<realv>(i,j)=value ;
    }
  }
  ifs >> temp;
  ifs >> temp;
  l.setWeights(weights);
  return ifs;
}
