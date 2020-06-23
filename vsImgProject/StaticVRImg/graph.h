#pragma once
#ifndef _GRAPH_H
#define _GRAPH_H
#define IGRAPH_STATIC 1
#include <opencv2/core.hpp>
#include <igraph.h>
namespace graph {

	struct igraph_init {
		static void attri_init();
		static bool status;
	};

	void graphTest();
	bool buildFull(igraph_t &graph, int n, bool directed = false);
	bool build(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts, igraph_t& graph);
}

#endif