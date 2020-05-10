#include "StaticVRImg/graph.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <numeric>
int main() {
	//graph::graphTest();
	//test 
	Eigen::MatrixXi test(9,2);
	test.col(0) << 1, 5, 6, 7, 9, 0, 10, 20, 11;
	test.col(1)	<< 3, 4, 6, 7, 8, 9, 0, 1, 5;
	Eigen::MatrixXi index(9,2);
	int x = 0;
	std::iota(index.col(0).data(), index.col(0).data()+index.col(0).size(), x++);
	int y = 0;
	std::iota(index.col(1).data(), index.col(1).data() + 9, y++);
	std::sort(index.col(0).data(), index.col(0).data()+9, [&](int left, int right) {return test.col(0)(left) < test.col(0)(right); });
	
	std::cout << "Orig :" << test << std::endl<<" indexs: ";
	std::cout << index;
	return 0;
}