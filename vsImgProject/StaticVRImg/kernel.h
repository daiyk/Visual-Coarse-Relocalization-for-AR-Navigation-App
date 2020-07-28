#pragma once
#ifndef _KERNEL_H
#define _KERNEL_H
#define IGRAPH_STATIC 1
#include <igraph.h>
#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Core>
namespace kernel {
	class covisMap {
	public:
		covisMap(int kCenters):kCenters(kCenters) {
			igraph_sparsemat_init(&this->map, kCenters, 1, kCenters * 500);
			this->inverted_tree.resize(kCenters);
		};

		//add labels in mygraph to inverted tree and covisibility map
		void process(igraph_t& mygraph);
		//retrieve the relevant graph ids and return it
		void processSampleLoc();
		std::vector <std::vector<int>> retrieve(igraph_t& queryGraph);
		void printMap();
		~covisMap() { igraph_sparsemat_destroy(&this->map); }
	private:
		std::vector<std::vector<int>> inverted_tree;
		igraph_sparsemat_t map;
		int kCenters;
	};


	class robustKernel{
	public:
		robustKernel(int h_max, size_t n_labels); //n_labels is used to build the neighborhood Vector
		~robustKernel();
		void push_back(igraph_t newgraph);
		void graphPrepro(igraph_t& graph);
		std::vector < std::vector<double>> robustKernelCompWithQueryArray(std::vector<igraph_t>& database_graphs);
		igraph_matrix_t robustKernelCom();
		double robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, igraph_t& graph_i, igraph_t& graph_j);

		//deprecated functions
		std::vector<std::vector<int>> raw_scores;
	private:
		size_t n_labels;
		//TODO:h_max is the iteration of relabeling, not necessary to be here, default to 1
		int h_max;
		std::set<int> label_sets;
		std::vector<size_t> ver_nums;
		std::vector<size_t> edge_nums;
		std::vector<igraph_t> graphs;
		std::vector<double> raw_self_kernel;
		std::vector<std::unordered_map<int, std::vector<size_t> > > inverted_indices;
		/*std::vector<unordered_map<int, vector<int> > > inverteds;*/
	};

	
}
#endif
