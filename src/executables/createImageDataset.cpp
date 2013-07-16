#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <stdio.h>
#include <fstream>

#include "../HnT.hpp"

using namespace std;
using namespace cv;

ImageDataset rimesLoader(string groundTruthFile, string groundTruthFolder) {
	string line, imageFile, label;
	size_t position;
	ifstream gtFile(groundTruthFile.c_str());
	ImageDataset dataset;
	if (gtFile.is_open()) {
		while (gtFile.good()) {
			getline(gtFile, line);
			position = line.find(" ");
			label = line.substr(position + 1);
			imageFile = groundTruthFolder + line.substr(0, position);
			dataset.addImage(imageFile);
		}
		gtFile.close();
	}
	return dataset;
}

int main(int argc, char* argv[]) {
	vector<string> arguments;
	arguments.push_back("ground truth file containing the location of the datasets");
	arguments.push_back("folder containing the datasets");
	arguments.push_back("name for the new file");
	cout << helper("Create Rimes Image Dataset", "Create an image dataset from Rimes files", arguments) << endl;
	if (argc != arguments.size() + 1) {
		cerr << "Not enough arguments, " << argc-1 << " given and "<< arguments.size()<<" required" << endl;
		return EXIT_FAILURE;
	}
	ImageDataset dataset;
	string groundTruthFile = argv[1];
	string groundTruthFolder = argv[2];
	string saveLocation = argv[3];

	
	dataset = rimesLoader(groundTruthFile, groundTruthFolder);
	dataset.save(saveLocation);
	return EXIT_SUCCESS;
}
