#include <string>
#include <iostream>
#include <vector>
#include "UKsets.h"
#include "StaticVRImg/fileManager.h"
void main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cout << "please provides the path to UKBench imagesets" << std::endl;
	}
	fileManager::read_user_set();
	std::cout << "center number: :" << fileManager::parameters::centers << std::endl;
	UKB::UKtrain(argc, argv);
	
	return;
	
}