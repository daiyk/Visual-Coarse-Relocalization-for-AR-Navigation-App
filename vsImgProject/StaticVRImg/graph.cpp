#include "graph.h"
#include "fileManager.h"
#include <vector>
#include <map>
#include <iostream>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>


using namespace cv;
using params = fileManager::parameters;

bool graph::igraph_init::status = false;
void graph::igraph_init::attri_init() {
	if (status == false) {
		igraph_i_set_attribute_table(&igraph_cattribute_table);
		status = true;
	}
}

void graph::graphTest() {
	igraph_init::attri_init();
	igraph_integer_t diameter; igraph_t graph;
	igraph_rng_seed(igraph_rng_default(), 42);
	igraph_erdos_renyi_game(&graph, IGRAPH_ERDOS_RENYI_GNP, 1000, 5.0 / 1000, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
	igraph_diameter(&graph, &diameter, 0, 0, 0, IGRAPH_UNDIRECTED, 1); 
	printf("Diameter of a random graph with average degree 5: %d\n", (int)diameter);
	igraph_destroy(&graph);
}

//build graph without any edge
//1th arg: macthes of visual words to features
//2th arg: all keypoints with opencv format
//3th arg: reference to non-initial igraph, final graph will be stored here
bool graph::buildEmpty(std::vector<DMatch>& matches, std::vector<KeyPoint>& kpts, igraph_t& mygraph) {
	igraph_init::attri_init();
	igraph_integer_t n_vertices = matches.size();
	igraph_bool_t loops = false;
	if (matches.size() == 0) {
		igraph_empty(&mygraph, 0, IGRAPH_UNDIRECTED);
		SETGAN(&mygraph, "n_vertices", 0);
		std::cout << "graph.build: warning: zero matches empty graph returned" << std::endl;
		return true;
	}

	SETGAS(&mygraph, "name", "kernelfullgraph");  // set graph name attribute
	SETGAN(&mygraph, "n_vertices", n_vertices); // set vertices number attribute
	int status = igraph_empty(&mygraph, n_vertices, IGRAPH_UNDIRECTED);

	//add attributes like distance, labels and so on
	if (status == IGRAPH_EINVAL) {
		std::cout << "build full graph: Invalid number of graph vertices" << std::endl;
		return false;
	}

	//define container for keypoints
	Eigen::MatrixXd pos(n_vertices, 2);
	//loop through and add edges
	for (size_t i = 0; i < n_vertices; i++) {
		pos.row(i) = Eigen::Vector2d(kpts[matches[i].queryIdx].pt.x, kpts[matches[i].queryIdx].pt.y);
	}

	//dynamic C array for labels and scales

	/*igraph_real_t *labels = (igraph_real_t*)malloc(sizeof(igraph_real_t)*n_vertices);
	igraph_real_t* scales = (igraph_real_t*)malloc(sizeof(igraph_real_t) * n_vertices); */// change it to std vector
	std::vector<igraph_real_t> labs(n_vertices), scls(n_vertices), posx(n_vertices), posy(n_vertices);
	igraph_real_t* labels = labs.data();
	igraph_real_t* scales = scls.data();
	igraph_real_t* positionx = posx.data();
	igraph_real_t* positiony = posy.data();
	//allocate word label to query nodes
	for (size_t i = 0; i < n_vertices; i++) {
		labs[i] = matches[i].trainIdx;
		scls[i] = kpts[matches[i].queryIdx].size;
		posx[i] = kpts[matches[i].queryIdx].pt.x;
		posy[i] = kpts[matches[i].queryIdx].pt.y;
	}
	//igraph add labels and other attributes 
	igraph_vector_t lab_vec, edge_vec, scale_vec, posx_vec, posy_vec;
	igraph_vector_view(&lab_vec, labels, n_vertices);
	igraph_vector_view(&scale_vec, scales, n_vertices);
	igraph_vector_view(&posx_vec, positionx, n_vertices);
	igraph_vector_view(&posy_vec, positiony, n_vertices);
	/*igraph_vector_init(&deg_vec, n_vertices);*/

	SETVANV(&mygraph, "label", &lab_vec);
	SETVANV(&mygraph, "scale", &scale_vec);
	SETVANV(&mygraph, "posx", &posx_vec);
	SETVANV(&mygraph, "posy", &posy_vec);

	igraph_attribute_combination_t comb;
	igraph_attribute_combination(&comb,
		"weight", IGRAPH_ATTRIBUTE_COMBINE_SUM,
		"", IGRAPH_ATTRIBUTE_COMBINE_FIRST,
		IGRAPH_NO_MORE_ATTRIBUTES);

	igraph_simplify(&mygraph, true, true, &comb);
	return true;
}

//build full graph
//NOTE: build full only keeps label attribute
//1th arg: macthes of visual words to features
//2th arg: all keypoints with opencv format
//3th arg: reference to non-initial igraph, final graph will be stored here
bool graph::buildFull(std::vector<DMatch>& matches, std::vector<KeyPoint>& kpts, igraph_t& mygraph,std::string graph_name) {
	igraph_init::attri_init();
	igraph_integer_t n_vertices = matches.size();
	igraph_bool_t loops = false;
	//no matching label find empty graph with 0 vertice return
	if (matches.size() == 0) {
		igraph_empty(&mygraph, 0, IGRAPH_UNDIRECTED);
		SETGAN(&mygraph, "n_vertices", 0);
		SETGAS(&mygraph, "name", "UNDEFINED");
		std::cout << "graph.build: warning: zero matches empty graph returned" << std::endl;
		return true;
	}

	int status = igraph_full(&mygraph, n_vertices, false, loops);
	SETGAS(&mygraph, "name", graph_name.c_str());  // set graph name attribute
	SETGAN(&mygraph, "n_vertices", n_vertices); // set vertices number attribute

	//add attributes like distance, labels and so on
	if (status == IGRAPH_EINVAL) {
		std::cout << "build full graph: Invalid number of graph vertices" << std::endl;
		return false;
	}

	////define container for keypoints
	//Eigen::MatrixXd pos(n_vertices, 2);
	//for (size_t i = 0; i < n_vertices; i++) {
	//	pos.row(i) = Eigen::Vector2d(kpts[matches[i].queryIdx].pt.x, kpts[matches[i].queryIdx].pt.y);
	//}

	std::vector<igraph_real_t> labs(n_vertices);
	igraph_real_t* labels = labs.data();
	//allocate word label to query nodes
	for (size_t i = 0; i < n_vertices; i++) {
		labs[i] = matches[i].trainIdx;
	}

	//igraph add labels attributes 
	igraph_vector_t lab_vec;
	igraph_vector_view(&lab_vec, labels, n_vertices);
	
	SETVANV(&mygraph, "label", &lab_vec);

	//igraph init weight attributes
	igraph_attribute_combination_t comb;
	igraph_attribute_combination(&comb,
		"weight", IGRAPH_ATTRIBUTE_COMBINE_SUM,
		"", IGRAPH_ATTRIBUTE_COMBINE_FIRST,
		IGRAPH_NO_MORE_ATTRIBUTES);
	igraph_simplify(&mygraph, true, true, &comb);
	int n_edges = igraph_ecount(&mygraph);
	std::vector<igraph_real_t> w(n_edges);
	igraph_real_t* weights = w.data();
	for (size_t i = 0; i < n_edges; i++) {
		w[i] = 1.0;
	}
	igraph_vector_t weight_vec;
	igraph_vector_view(&weight_vec, weights, n_edges);
	SETEANV(&mygraph, "weight", &weight_vec);
	return true;
}

//if matches is empty, a empty graph is returned
//mygraph must be a uninit graph
bool graph::build(std::vector<DMatch> &matches, std::vector<KeyPoint> &kpts, igraph_t &mygraph,std::string graph_name) {
	igraph_init::attri_init();
	//build from matches result and parameter setting
	if (matches.size() > kpts.size()) {
		std::cout << "graph.buld: matches size cannot larger than the keypoint size!" << std::endl;
		return false;
	}
	graph::igraph_init::attri_init();
	//to ensure the graph building process, for zero matching graphs return empty graph
	if (matches.size() == 0) {
		igraph_empty(&mygraph, 0, IGRAPH_UNDIRECTED);
		SETGAN(&mygraph, "n_vertices", 0);
		SETGAS(&mygraph, "name", "UNDEFINED");
		std::cout << "graph.build: warning: zero matches empty graph returned" << std::endl;
		return true;
	}
	clock_t sTime = clock();
	int n_vertices = matches.size();
	
	igraph_empty(&mygraph,n_vertices,IGRAPH_UNDIRECTED);
	SETGAS(&mygraph, "name", (graph_name+"Deg"+std::to_string(params::maxNumDeg)).c_str());  // set graph name attribute
	SETGAN(&mygraph, "n_vertices", n_vertices); // set vertices number attribute
	// add more information to the graph, like the vertices number and edge number
	//define container for keypoints and compute distance
	Eigen::MatrixXd pos(n_vertices, 2);
	Eigen::MatrixXd dists(n_vertices, n_vertices);
	//loop through and add edges
	for (size_t i = 0; i < n_vertices;i++) {
		pos.row(i) = Eigen::Vector2d(kpts[matches[i].queryIdx].pt.x, kpts[matches[i].queryIdx].pt.y);
	}

	for (size_t i = 0; i < n_vertices; i++) {
		//compute the distance norm and store on the matrix
		auto value = (pos.bottomRows(n_vertices - i).rowwise() - pos.row(i)).matrix().rowwise().norm(); // col vector
		dists.row(i).tail(n_vertices - i) = value.transpose();
		dists.col(i).tail(n_vertices - i) = value;
	}

	//sort the distance in ascend order for graph connection
	Eigen::MatrixXi indexes(n_vertices,n_vertices);

	//initialization with continuous numbers
	for (size_t i = 0; i < n_vertices; i++) {
		size_t x = 0;
		std::iota(indexes.col(i).data(), indexes.col(i).data()+n_vertices, x); // sequence assignment for index value
	}
	//sort the distances and stores the indexes, col represents the sequence! from small to large
	for (size_t i = 0; i < n_vertices; i++) {
		std::sort(indexes.col(i).data(), indexes.col(i).data() + n_vertices, [&](size_t left, size_t right) {return (dists.col(i))(left) < (dists.col(i))(right); });
	}
	
	//dynamic C array for labels and scales

	/*igraph_real_t *labels = (igraph_real_t*)malloc(sizeof(igraph_real_t)*n_vertices);
	igraph_real_t* scales = (igraph_real_t*)malloc(sizeof(igraph_real_t) * n_vertices); */// change it to std vector
	std::vector<igraph_real_t> labs(n_vertices);// scls(n_vertices), posx(n_vertices), posy(n_vertices);
	igraph_real_t* labels = labs.data();
	/*igraph_real_t* scales = scls.data();
	igraph_real_t* positionx = posx.data();
	igraph_real_t* positiony = posy.data();*/
	
	//allocate word label to query nodes
	for (size_t i = 0; i < n_vertices;i++) {
		labs[i] = matches[i].trainIdx;
		/*scls[i] = kpts[matches[i].queryIdx].size;
		posx[i] = kpts[matches[i].queryIdx].pt.x;
		posy[i] = kpts[matches[i].queryIdx].pt.y;*/
	}
	//igraph add labels and degrees attributes 
	igraph_vector_t lab_vec, edge_vec;// scale_vec, posx_vec, posy_vec;
	
	igraph_vector_view(&lab_vec, labels, n_vertices);
	/*igraph_vector_view(&scale_vec, scales, n_vertices);
	igraph_vector_view(&posx_vec, positionx, n_vertices);
	igraph_vector_view(&posy_vec, positiony, n_vertices);*/
	/*igraph_vector_init(&deg_vec, n_vertices);*/

	SETVANV(&mygraph, "label", &lab_vec);
	/*SETVANV(&mygraph, "scale", &scale_vec);
	SETVANV(&mygraph, "posx", &posx_vec);
	SETVANV(&mygraph, "posy", &posy_vec);*/
	/*SETVANV(&mygraph, "degree", &deg_vec);*/
	
	//add edge
	double radDegLim;
	if (params::radDegLim == -1) {
		radDegLim = std::numeric_limits<double>::max();
	}
	else
	{
		radDegLim = params::radDegLim;
	}
	std::vector<igraph_real_t> edges;
	edges.reserve(params::maxNumDeg * 2 *n_vertices);
	for (size_t i = 0; i < n_vertices; i++) {
		/* compare the distance and deg limits
		*   a. Here each vertex is limited on the number of their connected vertices
		*	the limiting number  = params::maxNumDeg
		*	b. the edge distance is constrained that edge should not exceed radDegLim * kpts.scale
		*/
		for (size_t j = 1; j < params::maxNumDeg+1 && j < n_vertices; j++) {
			if (dists(i, indexes(j, i)) < radDegLim){ //* VAN(&mygraph,"scale",i)) {
				edges.push_back(i);
				edges.push_back(indexes(j,i));
			}
		}
	}
	

	size_t n_edges = edges.size() / 2;
	
	igraph_vector_view(&edge_vec, edges.data(), edges.size());
	igraph_add_edges(&mygraph, &edge_vec, 0);
	SETGAN(&mygraph, "edges", igraph_ecount(&mygraph)); //set edge number attribute
	SETGAN(&mygraph, "degree", fileManager::parameters::maxNumDeg);

	//set edge weight attributes, can be deleted later
	std::vector<igraph_real_t> eweight(igraph_ecount(&mygraph), 1.0);
	igraph_real_t* edgeW = eweight.data();
	igraph_vector_t eweight_vec;
	igraph_vector_view(&eweight_vec, edgeW, igraph_ecount(&mygraph));
	SETEANV(&mygraph, "weight", &eweight_vec);

	igraph_attribute_combination_t comb;
	igraph_attribute_combination(&comb,
		"weight", IGRAPH_ATTRIBUTE_COMBINE_SUM,
		"", IGRAPH_ATTRIBUTE_COMBINE_FIRST,
		IGRAPH_NO_MORE_ATTRIBUTES);

	igraph_simplify(&mygraph, true, true, &comb);
	/*std::cout << " ->graph building spend " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec...." << std::endl;*/
	return true;
}

//source graph will be extended
//RANSAC method to extend the sourceGraph by merging extendGraph
//bestMatches should be the extendGraph as query to the sourceGraph
//best matches should be generated from the compressed descriptors matching!
//new graph stores in the sourcegraph
//keypoints matching
bool graph::extendDemo(igraph_t& sourceGraph, igraph_t& extendGraph, std::vector<DMatch> &bestMatches)
{
	if (bestMatches.size() < 8) {
		std::cout << "graph::extend: Error: opencv ransac need at least 8 points.";
		return false;
	}
	
	igraph_init();
	//separate the inliners and outliers
	int n_srcVertices = GAN(&sourceGraph, "n_vertices");
	int n_exdVertices = GAN(&extendGraph, "n_vertices");

	//extract keypoints from the exdgraph
	igraph_vector_t srcPosx, srcPosy, exdPosx, exdPosy,exdScale,exdLabel;
	igraph_vector_init(&srcPosx, 0);
	igraph_vector_init(&srcPosy, 0);
	igraph_vector_init(&exdPosy, 0);
	igraph_vector_init(&exdPosx, 0);
	igraph_vector_init(&exdScale, 0);
	igraph_vector_init(&exdLabel, 0);

	VANV(&sourceGraph, "posx", &srcPosx);
	VANV(&sourceGraph, "posy", &srcPosy);

	VANV(&extendGraph, "posx", &exdPosx);
	VANV(&extendGraph, "posy", &exdPosy);

	VANV(&extendGraph, "scale", &exdScale);
	VANV(&extendGraph, "label", &exdLabel);

	std::vector<cv::Point2f> srcKpts(bestMatches.size()), exdKpts(bestMatches.size()),allexdKpts(n_exdVertices);

	//matches is filtered and may not contains all points
	for (size_t i = 0; i < bestMatches.size(); i++) {
		exdKpts[i].x=VECTOR(exdPosx)[bestMatches[i].queryIdx];
		exdKpts[i].y = VECTOR(exdPosy)[bestMatches[i].queryIdx];
		srcKpts[i].x = VECTOR(srcPosx)[bestMatches[i].trainIdx];
		srcKpts[i].y = VECTOR(srcPosy)[bestMatches[i].trainIdx];
	}
	for (size_t i = 0; i < n_exdVertices; i++) {
		allexdKpts[i].x = VECTOR(exdPosx)[i];
		allexdKpts[i].y = VECTOR(exdPosy)[i];
	}
	//compute the homography matrix
	cv::Mat mask;
	cv::Mat homo = findHomography(exdKpts, srcKpts, mask, cv::RANSAC);

	//transform the graph points and rebuild the graph

	std::vector<Point2f> transfKpts;
	perspectiveTransform(allexdKpts, transfKpts, homo);

	std::vector<KeyPoint> mergeKpts;
	std::map<size_t,size_t> inliers, outliers;
	std::vector<size_t> outliers_idx;
	//find outliers and record them
	for (size_t i = 0; i < mask.rows; i++) {
		if (mask.at<uchar>(i)) {
			inliers.insert({ bestMatches[i].queryIdx,bestMatches[i].trainIdx });
		}
	}
	for (size_t i = 0; i < igraph_vcount(&extendGraph); i++) {
		//if not inside inliers then add to the container
		if (inliers.find(i) == inliers.end()) {
			outliers_idx.push_back(i);
			mergeKpts.push_back(KeyPoint(transfKpts[i], VECTOR(exdScale)[i], -1.0, 0,0, VECTOR(exdLabel)[i]));
		}
	}
	//default the new vertices are new continuous larger number
	igraph_add_vertices(&sourceGraph, outliers_idx.size(),0);

	//n_srcVertices is the original srcG v number, map the outliers from extend graph to the source graph
	for (size_t i = 0; i < outliers_idx.size(); i++) {
		outliers.insert({ outliers_idx[i] , n_srcVertices + i });
	}

	//check inliers and outliers, accmulate or add new edge weights
	for (size_t i = 0; i < igraph_vcount(&extendGraph); i++) {
		for (size_t j = i; j < igraph_vcount(&extendGraph); j++) {
			igraph_integer_t eid;
			igraph_get_eid(&extendGraph, &eid, i, j, IGRAPH_UNDIRECTED, false);
			if (eid == -1) {
				continue; //no edge found on (i,j) just skip
			}
			//if both of them are in inliers then do edge weights merging
			size_t map_i, map_j;
			auto weight = EAN(&extendGraph, "weight", eid);
			if (inliers.find(i) != inliers.end()) {
				map_i = inliers.at(i);
			}
			else
			{
				map_i = outliers.at(i);
			}
			if (inliers.find(j) != inliers.end())
			{
				map_j = inliers.at(j);
			}
			else
			{
				map_j = outliers.at(j);
			}
			igraph_integer_t srceid;
			igraph_get_eid(&sourceGraph, &srceid, map_i, map_j, IGRAPH_UNDIRECTED, false);
			if (srceid != -1) {
				//merge weight with original graph this is only applied for map_i,map_j both inliers
				SETEAN(&sourceGraph, "weight", srceid, EAN(&sourceGraph, "weight", srceid) + weight);
			}
			else
			{
				//add new edge and weight
				igraph_add_edge(&sourceGraph, map_i, map_j);
				igraph_get_eid(&sourceGraph, &srceid, map_i, map_j, IGRAPH_UNDIRECTED, false);
				SETEAN(&sourceGraph, "weight", srceid, weight);
			}

		}
	}
	//add outliers and its attris to the source graph
	for (size_t i = 0; i < outliers_idx.size(); i++) {
		SETVAN(&sourceGraph, "posx", outliers.at(outliers_idx[i]), mergeKpts[i].pt.x);
		SETVAN(&sourceGraph, "posy", outliers.at(outliers_idx[i]), mergeKpts[i].pt.y);
		SETVAN(&sourceGraph, "scale", outliers.at(outliers_idx[i]), mergeKpts[i].size);
		SETVAN(&sourceGraph, "label", outliers.at(outliers_idx[i]), mergeKpts[i].class_id);
	}
	//set new number of vertics
	SETGAN(&sourceGraph, "n_vertices", igraph_vcount(&sourceGraph));
	
	return true;
}

//the best matches should be queryId->extendGraph, trainId->sourceGraph
bool graph::extend(igraph_t& sourceGraph, igraph_t& extendGraph, std::vector<cv::KeyPoint>& srckpts, std::vector<cv::KeyPoint>& qrykpts, std::vector<cv::DMatch>& bestMatches)
{
	if (bestMatches.size() < 8) {
		std::cout << "graph::extend: Error: opencv ransac need at least 8 points.";
		return false;
	}

	igraph_i_set_attribute_table(&igraph_cattribute_table);
	//separate the inliners and outliers
	int n_srcVertices = GAN(&sourceGraph, "n_vertices");
	int n_exdVertices = GAN(&extendGraph, "n_vertices");

	//extract keypoints from the exdgraph
	igraph_vector_t srcPosx, srcPosy, exdPosx, exdPosy, exdScale, exdLabel;
	igraph_vector_init(&srcPosx, 0);
	igraph_vector_init(&srcPosy, 0);
	igraph_vector_init(&exdPosy, 0);
	igraph_vector_init(&exdPosx, 0);
	igraph_vector_init(&exdScale, 0);
	igraph_vector_init(&exdLabel, 0);

	VANV(&sourceGraph, "posx", &srcPosx);
	VANV(&sourceGraph, "posy", &srcPosy);

	VANV(&extendGraph, "posx", &exdPosx);
	VANV(&extendGraph, "posy", &exdPosy);

	VANV(&extendGraph, "scale", &exdScale);
	VANV(&extendGraph, "label", &exdLabel);

	std::vector<cv::Point2f> srcKpts(bestMatches.size()), exdKpts(bestMatches.size()), allexdKpts(n_exdVertices);

	//matches is filtered and may not contains all points
	for (size_t i = 0; i < bestMatches.size(); i++) {
		exdKpts[i].x = VECTOR(exdPosx)[bestMatches[i].queryIdx];
		exdKpts[i].y = VECTOR(exdPosy)[bestMatches[i].queryIdx];
		srcKpts[i].x = VECTOR(srcPosx)[bestMatches[i].trainIdx];
		srcKpts[i].y = VECTOR(srcPosy)[bestMatches[i].trainIdx];
	}
	for (size_t i = 0; i < n_exdVertices; i++) {
		allexdKpts[i].x = VECTOR(exdPosx)[i];
		allexdKpts[i].y = VECTOR(exdPosy)[i];
	}
	//compute the homography matrix
	cv::Mat mask;
	cv::Mat homo = findHomography(exdKpts, srcKpts, mask, cv::RANSAC);

	//transform the graph points and rebuild the graph

	std::vector<Point2f> transfKpts;
	perspectiveTransform(allexdKpts, transfKpts, homo);

	std::vector<KeyPoint> mergeKpts;
	std::map<size_t, size_t> inliers, outliers;
	std::vector<size_t> outliers_idx;

	//find outliers and record them
	for (size_t i = 0; i < mask.rows; i++) {
		if (mask.at<uchar>(i)) {
			inliers.insert({ bestMatches[i].queryIdx,bestMatches[i].trainIdx });
		}
	}
	for (size_t i = 0; i < igraph_vcount(&extendGraph); i++) {
		//if not inside inliers then add to the container
		if (inliers.find(i) == inliers.end()) {
			outliers_idx.push_back(i);
			mergeKpts.push_back(KeyPoint(transfKpts[i], VECTOR(exdScale)[i], -1.0, 0, 0, VECTOR(exdLabel)[i]));
		}
	}
	//default the new vertices are new continuous larger number
	igraph_add_vertices(&sourceGraph, outliers_idx.size(), 0);

	//n_srcVertices is the original srcG v number, map the outliers from extend graph to the source graph
	for (size_t i = 0; i < outliers_idx.size(); i++) {
		outliers.insert({ outliers_idx[i] , n_srcVertices + i });
	}

	//check inliers and outliers, accmulate or add new edge weights
	for (size_t i = 0; i < igraph_vcount(&extendGraph); i++) {
		for (size_t j = i; j < igraph_vcount(&extendGraph); j++) {
			igraph_integer_t eid;
			igraph_get_eid(&extendGraph, &eid, i, j, IGRAPH_UNDIRECTED, false);
			if (eid == -1) {
				continue; //no edge found on (i,j) just skip
			}
			//if both of them are in inliers then do edge weights merging
			size_t map_i, map_j;
			auto weight = EAN(&extendGraph, "weight", eid);
			if (inliers.find(i) != inliers.end()) {
				map_i = inliers.at(i);
			}
			else
			{
				map_i = outliers.at(i);
			}
			if (inliers.find(j) != inliers.end())
			{
				map_j = inliers.at(j);
			}
			else
			{
				map_j = outliers.at(j);
			}
			igraph_integer_t srceid;
			igraph_get_eid(&sourceGraph, &srceid, map_i, map_j, IGRAPH_UNDIRECTED, false);
			if (srceid != -1) {
				//merge weight with original graph this is only applied for map_i,map_j both inliers
				SETEAN(&sourceGraph, "weight", srceid, EAN(&sourceGraph, "weight", srceid) + weight);
			}
			else
			{
				//add new edge and weight
				igraph_add_edge(&sourceGraph, map_i, map_j);
				igraph_get_eid(&sourceGraph, &srceid, map_i, map_j, IGRAPH_UNDIRECTED, false);
				SETEAN(&sourceGraph, "weight", srceid, weight);
			}

		}
	}
	//add outliers and its attris to the source graph
	for (size_t i = 0; i < outliers_idx.size(); i++) {
		SETVAN(&sourceGraph, "posx", outliers.at(outliers_idx[i]), mergeKpts[i].pt.x);
		SETVAN(&sourceGraph, "posy", outliers.at(outliers_idx[i]), mergeKpts[i].pt.y);
		SETVAN(&sourceGraph, "scale", outliers.at(outliers_idx[i]), mergeKpts[i].size);
		SETVAN(&sourceGraph, "label", outliers.at(outliers_idx[i]), mergeKpts[i].class_id);
	}
	//set new number of vertics
	SETGAN(&sourceGraph, "n_vertices", igraph_vcount(&sourceGraph));

	return true;
}

//bestMatches should be query_id(extend graph) to train_id(sourceGraph)
bool graph::extend(igraph_t& sourceGraph, igraph_t& extendGraph, std::vector<cv::DMatch>& bestMatches) {
	igraph_init();
	if (bestMatches.size() == 0) {
		return false;
	}
	std::vector < igraph_real_t > new_edge_ids;
	//source graph vertices number
	int n_srcVertices = GAN(&sourceGraph, "n_vertices");

	//extented graph vertices number
	int n_exdVertices = GAN(&extendGraph, "n_vertices");

	if (n_srcVertices == 0 || n_exdVertices == 0) {
		return false;
	}
	//igraph vector to store src and exd graph nodel label
	std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> srcLabel(new igraph_vector_t(), &igraph_vector_destroy), 
		exdLabel(new igraph_vector_t(), &igraph_vector_destroy);
	igraph_vector_init(srcLabel.get(), 0);
	igraph_vector_init(exdLabel.get(), 0);

	VANV(&sourceGraph, "label", srcLabel.get());
	VANV(&extendGraph, "label", exdLabel.get());

	/*std::vector<int> labels, labele;
	for (int k = 0; k < igraph_vector_size(srcLabel.get()); k++) {
		labels.push_back(VECTOR(*srcLabel)[k]);
	}
	for (int k = 0; k < igraph_vector_size(exdLabel.get()); k++) {
		labele.push_back(VECTOR(*exdLabel)[k]);
	}*/

	//count the total number of vertices of the new graph
	int n_mergedVectirces = n_srcVertices + n_exdVertices - bestMatches.size();
	//reserve space for new edge ids
	new_edge_ids.reserve((n_exdVertices-bestMatches.size())*(n_exdVertices-1));

	//merge graphs to the source graph
	igraph_add_vertices(&sourceGraph, n_mergedVectirces - n_srcVertices, 0);
	//inliers map that stores the corresponding vertices of the two graphs
	std::unordered_map<int, int> inliers, outliers;
	for (int i = 0; i < bestMatches.size(); i++) {
		inliers.insert({ bestMatches[i].queryIdx,bestMatches[i].trainIdx }); //queryId = exdGraph
	}

	//build outliers map
	int counter = 0;
	for (int i = 0; i < n_exdVertices; i++) {
		if (inliers.find(i) == inliers.end()) { //cannot find the queryId = exdGraph_node_Id
			outliers.insert({ i,n_srcVertices + counter });
			counter++;
		}
	}
	//check inliers, accmulate and merge edge weight
	for (int i = 0; i < n_exdVertices-1; i++) {
		for (int j = i+1; j < n_exdVertices; j++) {
			//igraph_integer_t eid;
			//igraph_get_eid(&extendGraph, &eid, i, j, IGRAPH_UNDIRECTED, false);
			////no edge found then continue
			//if (eid == -1) {
			//	continue;
			//}
			igraph_bool_t connected;
			igraph_are_connected(&extendGraph, i, j, &connected);
			if (!connected) {
				continue;
			}
			int map_i, map_j; //mapped exdGraph index in mergedGraph 
			/*auto weight = EAN(&extendGraph, "weight", eid);*/
			if (inliers.find(i) != inliers.end()) {
				map_i = inliers.at(i);
			}
			else
			{
				map_i = outliers.at(i);
			}
			if (inliers.find(j) != inliers.end())
			{
				map_j = inliers.at(j);
			}
			else
			{
				map_j = outliers.at(j);
			}
			//check Edge_ij existence
			/*igraph_integer_t srcEid;
			igraph_bool_t connected;
			igraph_are_connected(&sourceGraph,map_i,map_j,&connected);*/
			new_edge_ids.push_back(map_i);
			new_edge_ids.push_back(map_j);
			//if (connected) {
			//	igraph_get_eid(&sourceGraph, &srcEid, map_i, map_j, IGRAPH_UNDIRECTED, false);
			//	//means the edge is inliers edge, merge weight with source graph 
			//	SETEAN(&sourceGraph, "weight", srcEid, EAN(&sourceGraph, "weight", srcEid) + weight);
			//}
			//else {
			//	//add new edge and weight
			//	igraph_add_edge(&sourceGraph, map_i, map_j);
			//	igraph_integer_t newEid;
			//	igraph_get_eid(&sourceGraph, &newEid, map_i, map_j, IGRAPH_UNDIRECTED, false);
			//	SETEAN(&sourceGraph, "weight", newEid, weight);
			//}
		}
	}

	//add edges for the new generated graph
	int old_edges_num = igraph_ecount(&sourceGraph);
	/*igraph_vector_t old_weights;
	igraph_vector_init(&old_weights,0);
	EANV(&sourceGraph, "weight", &old_weights);*/

	igraph_vector_t igraph_new_edges;
	igraph_vector_view(&igraph_new_edges, new_edge_ids.data(), new_edge_ids.size());
	igraph_add_edges(&sourceGraph, &igraph_new_edges, 0);

	//set the new vertices and add attributes for the outliers
	igraph_vector_t gtypes, vtypes, etypes;
	igraph_strvector_t gnames, vnames, enames;

	igraph_vector_init(&gtypes, 0);
	igraph_vector_init(&vtypes, 0);
	igraph_vector_init(&etypes, 0);

	igraph_strvector_init(&gnames, 0);
	igraph_strvector_init(&vnames, 0);
	igraph_strvector_init(&enames, 0);

	igraph_cattribute_list(&extendGraph, &gnames, &gtypes, &vnames, &vtypes, &enames, &etypes);
	
	//iterate through the vertex attribuets and assign the attributes to the vertex
	
	for (int j = 0; j < igraph_strvector_size(&vnames); j++) {
		if (VECTOR(vtypes)[j] == IGRAPH_ATTRIBUTE_NUMERIC){ 
			for (auto& i : outliers) {
				SETVAN(&sourceGraph, STR(vnames, j), i.second, VAN(&extendGraph, STR(vnames, j), i.first));
				/*SETVAN(&sourceGraph, "label", i.second, VECTOR(*exdLabel)[i.first]);*/
			}
		}
	}
		

	//set new sourceGraph property
	SETGAN(&sourceGraph, "n_vertices", igraph_vcount(&sourceGraph));

	//set the new weight
	for (int i = old_edges_num; i < igraph_ecount(&sourceGraph); i++) {
		SETEAN(&sourceGraph, "weight", i, 1.0);
	}
	std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> edge_weight(new igraph_vector_t(), &igraph_vector_destroy);
	igraph_vector_init(edge_weight.get(), 0);
	EANV(&sourceGraph, "weight", edge_weight.get());
	if (igraph_vector_size(edge_weight.get()) != igraph_ecount(&sourceGraph)) {
		std::cerr << "graph.extend: error: not all edge has \"weight\" attributes!\n";
		return false;
	}
	
	igraph_attribute_combination_t comb;
	igraph_attribute_combination(&comb,
		"weight", IGRAPH_ATTRIBUTE_COMBINE_SUM,
		"", IGRAPH_ATTRIBUTE_COMBINE_FIRST,
		IGRAPH_NO_MORE_ATTRIBUTES);
	igraph_simplify(&sourceGraph, 1, 1, &comb);
	igraph_attribute_combination_destroy(&comb);
	return true;
}

bool graph::extend1to2(igraph_t& sourceGraph, igraph_t& extendGraph, std::vector <cv::DMatch>& matches) {
	//extract attributes for future assignments
	igraph_init();
	if (matches.size() == 0) {
		std::cerr << "graph.extend1to2: matches size is zero, invalid extention! return source graph instead!\n";
		return true;
	}
	int n_source_vertices = GAN(&sourceGraph, "n_vertices");
	int n_exd_vertices = GAN(&extendGraph, "n_vertices");
	std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> 
		source_label(new igraph_vector_t(), &igraph_vector_destroy),
		exd_label(new igraph_vector_t(), &igraph_vector_destroy),
		source_edge_weight(new igraph_vector_t(), &igraph_vector_destroy),
		exd_edge_weight(new igraph_vector_t(), &igraph_vector_destroy);

	igraph_vector_init(source_label.get(),0);
	igraph_vector_init(exd_label.get(), 0);
	igraph_vector_init(source_edge_weight.get(), 0);
	igraph_vector_init(exd_edge_weight.get(), 0);
	VANV(&sourceGraph, "label",source_label.get());
	VANV(&extendGraph, "label", exd_label.get());
	EANV(&sourceGraph, "weight", source_edge_weight.get());
	EANV(&extendGraph, "weight", exd_edge_weight.get());

	//union of two graph
	igraph_t union_graph;
	igraph_disjoint_union(&union_graph, &sourceGraph, &extendGraph);

	//reset the attributes
	igraph_vector_append(source_label.get(), exd_label.get());
	igraph_vector_append(source_edge_weight.get(), exd_edge_weight.get());
	SETVANV(&union_graph, "label", source_label.get());
	SETEANV(&union_graph, "weight", source_edge_weight.get());

	//set attributes back
	//set the new vertices and add attributes for the outliers
	//igraph_vector_t gtypes, vtypes, etypes;
	//igraph_strvector_t gnames, vnames, enames;
	//igraph_vector_init(&gtypes, 0);
	//igraph_vector_init(&vtypes, 0);
	//igraph_vector_init(&etypes, 0);
	//igraph_strvector_init(&gnames, 0);
	//igraph_strvector_init(&vnames, 0);
	//igraph_strvector_init(&enames, 0);
	//igraph_cattribute_list(&sourceGraph, &gnames, &gtypes, &vnames, &vtypes, &enames, &etypes);
	////iterate through the vertex attribuets and assign the attributes to the vertex
	//for (int j = 0; j < igraph_strvector_size(&vnames); j++) {
	//	if (VECTOR(vtypes)[j] == IGRAPH_ATTRIBUTE_NUMERIC) {
	//		for (int i = 0;i<n_source_vertices;i++) {
	//			SETVAN(&union_graph, STR(vnames, j), i, VAN(&sourceGraph, STR(vnames, j), i));
	//			/*SETVAN(&sourceGraph, "label", i.second, VECTOR(*exdLabel)[i.first]);*/
	//		}
	//		for (int i = 0; i < n_exd_vertices; i++) {
	//			SETVAN(&union_graph, STR(vnames, j), i+n_source_vertices, VAN(&extendGraph, STR(vnames, j), i));
	//			/*SETVAN(&sourceGraph, "label", i.second, VECTOR(*exdLabel)[i.first]);*/
	//		}
	//	}
	//}

	igraph_vector_t	contract_vertices_list;
	std::vector<igraph_real_t> vertices_ids(igraph_vcount(&union_graph),-1);
	for (int k = 0; k < matches.size(); k++) {
		vertices_ids[matches[k].queryIdx + n_source_vertices] = matches[k].trainIdx;
	}
	int count = 0;
	for (int i = 0; i < vertices_ids.size(); i++) {
		if (vertices_ids[i] == -1.0) {
			vertices_ids[i] = count;
			count++;
		}
	}

	//change the mapping vertices id: query: exd, train: source
	igraph_vector_view(&contract_vertices_list, vertices_ids.data(), vertices_ids.size());
	//start contraction
	igraph_attribute_combination_t comb;
	igraph_attribute_combination(&comb,
		"weight", IGRAPH_ATTRIBUTE_COMBINE_SUM,
		"", IGRAPH_ATTRIBUTE_COMBINE_FIRST,
		IGRAPH_NO_MORE_ATTRIBUTES);
	igraph_contract_vertices(&union_graph, &contract_vertices_list, &comb);
	igraph_simplify(&union_graph, 1, 1, &comb);
	SETGAN(&union_graph, "n_vertices", igraph_vcount(&union_graph));
	/*std::string new_name = std::string(GAS(&sourceGraph, "name")) + "/" + std::string(GAS(&extendGraph, "name"));*/
	SETGAS(&union_graph, "name", GAS(&sourceGraph, "name"));
	igraph_destroy(&sourceGraph);
	sourceGraph = union_graph; //shallow copy? deep copy?
}
