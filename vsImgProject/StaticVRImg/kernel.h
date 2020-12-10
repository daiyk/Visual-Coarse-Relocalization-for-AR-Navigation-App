#pragma once
#ifndef _KERNEL_H
#define _KERNEL_H
#define IGRAPH_STATIC 1
#include <igraph.h>
#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Core>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <boost/dynamic_bitset.hpp>
#include <feature/types.h>
namespace kernel {
	using scoreType = std::unordered_map<int, std::vector<float>>;
	using vetIndType = std::unordered_map<int, std::vector<size_t>>; //inverted index type
	class covisMap {
	public:

		covisMap(int kCenters, int dbSize = 1000):numCenters_(kCenters), numDBImgs_(dbSize) {
			igraph_sparsemat_init(&this->map, kCenters, 1, kCenters * 500);
			this->inverted_tree.resize(numCenters_);
			this->invert_index_.resize(numDBImgs_+1); //because imageId is start with 1
		};
		~covisMap() { igraph_sparsemat_destroy(&this->map); }
		//add labels in mygraph to inverted tree and covisibility map
		void process(igraph_t& mygraph);

		//build inverted_index tree
		void addEntry(int image_id, colmap::FeatureVisualIDs& id);

		//check the existence of id statistics with image_id
		bool existEntry(int image_id);

		//get visual id statistics of given image_id
		boost::dynamic_bitset<uint8_t> getEntry(int image_id);
		
		//query the map and return the best candidates
		void Query(colmap::FeatureVisualIDs& qryId, std::vector<int>& candids);

		//resize the covisMap
		void resize(int new_size) { if (new_size > invert_index_.size()) invert_index_.resize(new_size); }

		//retrieve the relevant graph ids and return it
		void processSampleLoc();
		std::vector <std::vector<int>> retrieve(igraph_t& queryGraph);
		void printMap();	
	private:
		std::vector<std::vector<int>> inverted_tree;
		std::vector<boost::dynamic_bitset<uint8_t>> invert_index_;
		igraph_sparsemat_t map;
		int numCenters_;
		int numDBImgs_;
	};


	class robustKernel{
	public:
		robustKernel(int h_max, size_t n_labels); //n_labels is used to build the neighborhood Vector
		~robustKernel();
		void push_back(igraph_t newgraph, int doc_ind = -1);
		void graphPrepro(igraph_t& graph, int doc_ind);
		std::vector < std::vector<double>> robustKernelCompWithQueryArray(std::vector<igraph_t>& database_graphs, std::vector<int>* indexes=nullptr, std::vector<int>* source_index=nullptr, bool tfidfWeight = false);
		igraph_matrix_t robustKernelCom();
		double robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, igraph_t& graph_i, igraph_t& graph_j, int doc_ind = -1);
		std::vector<igraph_t>& getGraphs() { return this->graphs; };
		void setTFIDF(cv::Mat& setTfidf) { tfidf = setTfidf;};
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
		cv::Mat tfidf;
		std::vector<std::unordered_map<int, std::vector<size_t> > > inverted_indices;
		/*std::vector<unordered_map<int, vector<int> > > inverteds;*/
	};

	class recurRobustKel {
	public:
		recurRobustKel(int h_max, size_t n_labels); //h_max is the number of max iteration, n_labels is the length of dict
		~recurRobustKel();
		void robustKernelCom(int i, igraph_t& source_graph, scoreType& kernel_vals);
		void robustKernelCom(int i, int j, scoreType &scores, vetIndType& inv1, vetIndType& inv2);
		void robustKernelCom(int i, igraph_t& source_graph, scoreType& scores, vetIndType& inv1, vetIndType& inv2, bool useTFIDF = false);
		void robustKernelCom(igraph_t& query_graph, igraph_t& source_graph, scoreType& kernel_vals, vetIndType& inv1, vetIndType& inv2, bool useTFIDF=false);
		double robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, igraph_t& graph_i, igraph_t& graph_j, int doc_ind = -1);

		std::vector < std::vector<float>> robustKernelCompWithQueryArray(std::vector<igraph_t>& database_graphs, std::vector<int>* source_indexes=nullptr);
		void push_back(igraph_t newgraph);
		void graphPrepro(igraph_t& graph);
		std::vector<igraph_t>& getGraphs() { return this->graphs; };
		void setTFIDF(cv::Mat& setTfidf) { tfidf = setTfidf; };
	private:
		void clearDatabaseGraphs(int n_query_graphs);
		int h_max;
		size_t n_labels;
		std::set<int> label_sets;
		std::vector<size_t> ver_nums;
		std::vector<size_t> edge_nums;
		std::vector<igraph_t> graphs;
		std::vector<double> raw_self_kernel;
		std::vector<std::unordered_map<int, std::vector<size_t> > > inverted_indices;
		cv::Mat tfidf;
	};

	
}
#endif
