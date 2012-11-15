#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>

#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]) {
	ifstream ifs(argv[1]);
	char temp[512];
	int numberOfFolds = atoi(argv[2]);
	int currentFold = 0;
	vector<ofstream*> outputs;
	for (int i = 0; i < numberOfFolds; i++) {
		ostringstream ss;
		ss << "fold" << (i+1) << ".txt";
		outputs.push_back(new ofstream(ss.str().c_str()));
	}
	while (ifs.good()) {
		ifs.getline(temp, 512);
		(*outputs[currentFold])<< string(temp) << endl;
		currentFold ++;
		if(currentFold>=numberOfFolds){
			currentFold = 0;
		}
	}
	for (int i = 0; i < numberOfFolds; i++) {
		outputs[i]->close();
		delete outputs[i];
	}
	return EXIT_SUCCESS;
}
