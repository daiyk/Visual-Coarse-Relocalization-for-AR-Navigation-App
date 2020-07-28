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
	bool buildEmpty(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts, igraph_t& mygraph);
	bool buildFull(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts, igraph_t& mygraph);
	bool build(std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kpts, igraph_t& graph);
	bool extend(igraph_t& sourceGraph, igraph_t& extendGraph, std::vector<cv::DMatch>& bestMatches);

}

#endif