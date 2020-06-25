#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Core>
#include "UKsets.h"
#include "Graphtest.h"
#include "StaticVRImg/fileManager.h"
#include "StaticVRImg/matcher.h"
#include "StaticVRImg/graph.h"
void main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cout << "please provides the path to UKBench imagesets" << std::endl;
	}
	fileManager::read_user_set();
	std::cout << "center number: :" << fileManager::parameters::centers << std::endl;


	//need to test the model for picture
	/*UKB::UKtrain(argc, argv, 1000);*/
	if (argc == 3) {
		UKB::UKtest(argc, argv, 80, 1000);
	}
	else if(argc==4)
	{
		graphBuildPlusKernelTest(argc, argv);
	}
	return;
	
}