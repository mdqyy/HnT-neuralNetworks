/*!
 * \file Machine.cpp
 * Body of the Machine class.
 * \author Luc Mioulet
 */

#include "Machine.hpp"

using namespace std;

Machine::Machine() : name("machine"){

}

Machine::Machine(std::string _name) : name(_name){

}

string Machine::getName() const{
  return name;
}

void Machine::setName(string _name){
  name=_name;
}

Machine::~Machine(){

}

ostream& operator<<(ostream& _os, const Machine& _m){
  _m.print(_os);
  return _os;
}
