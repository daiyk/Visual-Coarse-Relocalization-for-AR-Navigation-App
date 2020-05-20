#pragma once
#ifndef _CLUSTER_H
#define _CLUSTER_H
#define IGRAPH_STATIC 1
#include <igraph.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
namespace kernel {
	class robustKernel{
	public:
		robustKernel(int h_max, size_t n_labels);
		~robustKernel();
		void push_back(igraph_t &newgraph);
		void graphPrepro(igraph_t& graph);
		igraph_matrix_t robustKernelCom();
		double robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, int i, int j);

		//deprecated functions
		static double kernelValue(const std::vector<int>& map1, const std::vector<int>& map2, int& i, int& j, std::vector<int>& num_v, Eigen::MatrixXd& node_nei);
		static void wlRobustKernel(std::vector<Eigen::MatrixXi>& E, std::vector<std::vector<int>>& V_label, std::vector<int>& num_v, std::vector<int>& num_e, int h_max, Eigen::MatrixXd& K_mat);
		
	private:
		size_t n_labels;
		int h_max;
		std::vector<size_t> ver_nums;
		std::vector<size_t> edge_nums;
		std::vector<igraph_t> graphs;
		std::vector<std::unordered_map<int, std::vector<size_t> > > inverted_indices;
		/*std::vector<unordered_map<int, vector<int> > > inverteds;*/
	};
}
#endif
