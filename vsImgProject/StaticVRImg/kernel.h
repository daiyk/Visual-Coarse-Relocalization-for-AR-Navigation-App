#pragma once
#ifndef _CLUSTER_H
#define _CLUSTER_H
#include <iostream>
#include <vector>
#include <Eigen/Core>
namespace kernel {
	double kernelValue(const std::vector<int>& map1, const std::vector<int>& map2, int& i, int& j, std::vector<int>& num_v, Eigen::MatrixXd& node_nei);
	void wlRobustKernel(std::vector<Eigen::MatrixXi>& E, std::vector<std::vector<int>>& V_label, std::vector<int>& num_v, std::vector<int>& num_e, int h_max, Eigen::MatrixXd& K_mat);
}
#endif
