#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Core>
#define IGRAPH_STATIC 1
#include <igraph.h>
#include <opencv2/core/eigen.hpp>
#include "UKsets.h"
#include "Graphtest.h"
#include "featureTest.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/graph.h"
#include "StaticVRImg/vlad.h"
#include "StaticVRImg/probModel.h"
using params = fileManager::parameters;

void vladTrain(std::string path) {
	std::string testFolder = path;
	std::vector<std::string> paths;
	UKB::UKBench ukb(testFolder, 1000);
	for (int i = 0; i < ukb.imgPaths.size(); i++) {
		for (int j = 0; j < ukb.imgPaths[i].size(); j++) {
			paths.push_back(ukb.imgPaths[i][j]);
		}
	}

	//train the vlad 
	vlad::vlad vladTrain(paths);
}


int main(int argc, const char* argv[]) {
	fileManager::read_user_set();
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	/*std::cout << "argument number: " << argc<<"\n";*/
	/*featureExtTest();
	return 0;*/
	if (argc < 2) {
		//std::cout << "please provides the path to UKBench imagesets" << std::endl;
		graphExtendTest();
		/*covdetTest();*/
		/*autoTest();*/
		/*probModel::databaseManager db("E:\\datasets\\gerrard-hall\\gerrard-hall\\database.db");
		db.testFunction();*/
		//recurKernelTest();
		
	}
	std::cout << "center number: " << params::centers << std::endl;

	if (argc == 2) {
		
		/*vladTrain(argv[1]);*/
		/*UKB::UKFLANNTest(argc, argv, params::sampleSize, params::imgsetSize);*/
		/*UKB::UKtrain(argc,argv,1000);*/
		std::string imgPath = argv[1];
		vocabReadTest(imgPath);

		/*databaseTest();*/
		return 0;
	}
	if (argc == 3) {
		/*UKB::UKRecurTest(argc, argv, params::sampleSize, params::imgsetSize);*/
		/*UKB::UKtest(argc, argv, params::sampleSize, params::imgsetSize);*/
		//test two imgs
		/*recurKernelTestWithImage(argc, argv);*/
		
		nhhdGraphTest(argv);
		/*flanntest(qry_imgs, argv[2]);*/

	}
	if(argc==4)
	{
		/*graphBuildPlusKernelTest(argc, argv);*/
		/*graphbuildTest(argc, argv);*/
		/*UKB::UKVladTest(argc, argv, params::sampleSize, params::imgsetSize);*/
		/*UKB::UKVladTest(argc, argv, params::sampleSize, params::imgsetSize);*/
	}
	return 0;

}