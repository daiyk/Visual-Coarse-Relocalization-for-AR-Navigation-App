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
#include "StaticVRImg/vlad.h"
int main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cout << "please provides the path to UKBench imagesets" << std::endl;
		return 0;
	}
	fileManager::read_user_set();
	std::cout << "center number: " << fileManager::parameters::centers << std::endl;

	//cv::Mat1f a(50, 50, 1.0);
	//int rows = a.rows;
	//for (int i = 0; i * 10 < rows; i++) {
	//	cv::Mat temp(a, cv::Range(i * 10, (i + 1) * 10));
	//	std::cout << "the "<<i<<" th submatrix"<<std::endl<<temp << std::endl;
	//}
	//need to test the model for picture
	/*UKB::UKtrain(argc, argv, 1000);*/
	/*if (argc == 2) {
		std::vector<std::string> paths;
		fileManager::read_files_in_path(argv[1], paths);
		vlad::vlad encoder(paths);
		auto enc = encoder.getEncoder();

		std::cout << "size is: " << enc.size()<<std::endl;	
	}*/
	if (argc == 3) {
		UKB::UKtest(argc, argv, 2, 20);
	}
	else if(argc==4)
	{
		/*graphBuildPlusKernelTest(argc, argv);*/
		
		UKB::UKVladTest(argc, argv, 80, 1000);
	}
	return 0;
	
}