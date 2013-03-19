#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>

#include "../HnT.hpp"


using namespace std;
using namespace cv;


int main (int argc, char* argv[]){
  /*RegressionDataset dataset;
  dataset.load("../xml/rimes-cv/Folds1/Training1.xml");*/
  /*  RegressionDataset dataset2;
    dataset2.simpleLoad("test.xml");
    cout << dataset2 << endl;*/
//  Mat testMat(5,1,CV_64FC1,1.0);
//  Mat meanMat(5,1,CV_64FC1,0.0);
//  Mat stdevMat(5,1,CV_64FC1,1.0);
//  FeatureVector testFv(testMat);
//  ValueVector mean(meanMat);
//  ValueVector stdev(stdevMat);
//  InputLayer il(5,mean,stdevMat);
//  InputLayer ilSig(5,mean,stdevMat);
//  InputLayer ilRec(5,mean,stdevMat);
//   //cout << testFv ;
//  LayerTanh th(2);
//  LayerSigmoid sig(2);
//  LayerSigmoid sigRec(2,"rec",true);
//  //cout << th ;
//  Connection c(&il,&th);
//  Connection coInToSig(&ilSig,&sig);
//  Connection coInToRecSig(&ilRec,&sigRec);
//  //cout << th ;
//  //cout << il;
//  //cout << c;
//  il.forward(testFv);
//  //cout << th.getOutputSignal();
//  //cout << dataset;
//  dataset.save("../xml/test-copy.xml");
//  RegressionDataset regset;
//  regset.load("../xml/testRegression.xml");
//  //cout << regset;
//  regset.save("../xml/saveReg.xml");
//
//  /* Test unsupervised */
//  UnsupervisedDataset usset;
//  usset.load("../xml/testUnsupervised.xml");
//  //cout << usset;
//  usset.save("../xml/saveUS.xml");
//
//  /* Test copy constructor */
//  InputLayer cil(il);
//  //cout << il << cil;
//  LayerTanh cth(th);
//  //cout << th << cth;
//  Connection cc(c);
//  //cout << c << cc;
//
//  /* Test clones */
//  InputLayer clil = *il.clone();
//  clil.setName("CloneIL");
//  //cout << " Clones" <<endl ;
//  //cout << clil;
//  Connection clc = *c.clone();
//  //cout << clc;
//  LayerTanh clth = *th.clone();
//  //cout << clth;
//  //cout << th.clone()->getName() << endl;
//  clth.setName("CloneTh");
//
//  LayerTanh t = LayerTanh(3,"test");
//  LayerTanh* test = &t;
//  LayerTanh* copyTest = test->clone();
//  //cout << test <<" "<<copyTest;
//
//  /* Cloning and linking */
//  //cout << endl << endl;
//  clil.setOutputConnection(&clc);
//  clc.setInputLayer(&clil);
//  clc.setOutputLayer(&clth);
//  clth.setInputConnection(&clc);
//  //cout << clil << clc << clth;
//  //cout << il << c << th ;
//
//
//  /* Loading and saving */
//  //cout << endl <<"Loading and saving" << endl;
//  ofstream out("test.txt");
//  out << mean << stdev << il << c;
//  //cout << "saving done" << endl;
//  ifstream in("test.txt");
//  ValueVector mean2,stdev2;
//  InputLayer il2;
//  Connection c2;
//  in >> mean2 >> stdev2 >> il2 >> c2;
//  //cout << "After loading "<< endl << mean2 << stdev2 << il2 << c2;
//
//  cout << "Test with recurrency" << endl;
//  ilSig.forward(testFv);
//  cout << sig.getOutputSignal() << endl;
//  ilSig.forward(testFv);
//  cout <<" Evolution of recurrent layer : "<< endl;
//  cout << sig.getOutputSignal() << endl;
//  ilRec.forward(testFv);
//  cout << sigRec.getOutputSignal() << endl;
//  ilSig.forward(testFv);
//  ilRec.forward(testFv);
//  cout << sigRec.getOutputSignal() << endl;
//  ilRec.forward(testFv);
//  ilRec.forward(testFv);
//  ilRec.forward(testFv);
//  cout << sigRec.getOutputSignal() << endl;
//  ilRec.forward(testFv);
//  ilRec.forward(testFv);
//  ilRec.forward(testFv);
//  cout << sigRec.getOutputSignal() << endl;
//  cout << sig.getOutputSignal() << endl;
//
//
//  /*delete &clc;
//  delete &clth;
//  delete &clil;*/
  return EXIT_SUCCESS;
}
