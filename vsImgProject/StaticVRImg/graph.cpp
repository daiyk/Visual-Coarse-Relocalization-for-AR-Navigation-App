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
	igraph_integer_t diameter; igraph_t graph;
	igraph_rng_seed(igraph_rng_default(), 42);
	igraph_erdos_renyi_game(&graph, IGRAPH_ERDOS_RENYI_GNP, 1000, 5.0 / 1000, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
	igraph_diameter(&graph, &diameter, 0, 0, 0, IGRAPH_UNDIRECTED, 1); 
	printf("Diameter of a random graph with average degree 5: %d\n", (int)diameter);
	igraph_destroy(&graph);
}

bool graph::buildEmpty(std::vector<DMatch>& matches, std::vector<KeyPoint>& kpts, igraph_t& mygraph) {
	igraph_integer_t n_vertices = matches.size();
	igraph_bool_t loops = false;
	if (matches.size() == 0) {
		igraph_empty(&mygraph, 0, IGRAPH_UNDIRECTED);
		SETGAN(&mygraph, "vertices", 0);
		std::cout << "graph.build: warning: zero matches empty graph returned" << std::endl;
		return true;
	}

	SETGAS(&mygraph, "name", "kernelfullgraph");  // set graph name attribute
	SETGAN(&mygraph, "vertices", n_vertices); // set vertices number attribute
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

	igraph_simplify(&mygraph, true, true, 0);
	return true;
}

//build full graph and return it
bool graph::buildFull(std::vector<DMatch>& matches, std::vector<KeyPoint>& kpts, igraph_t& mygraph) {
	igraph_integer_t n_vertices = matches.size();
	igraph_bool_t loops = false;
	if (matches.size() == 0) {
		igraph_empty(&mygraph, 0, IGRAPH_UNDIRECTED);
		SETGAN(&mygraph, "vertices", 0);
		std::cout << "graph.build: warning: zero matches empty graph returned" << std::endl;
		return true;
	}

	SETGAS(&mygraph, "name", "kernelfullgraph");  // set graph name attribute
	SETGAN(&mygraph, "vertices", n_vertices); // set vertices number attribute
	int status = igraph_full(&mygraph, n_vertices, false, loops);

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
	std::vector<igraph_real_t> labs(n_vertices), scls(n_vertices), posx(n_vertices), posy(n_vertices),eweight(igraph_ecount(&mygraph), 1);
	igraph_real_t* labels = labs.data();
	igraph_real_t* scales = scls.data();
	igraph_real_t* positionx = posx.data();
	igraph_real_t* positiony = posy.data();
	igraph_real_t* edgeW = eweight.data();
	//allocate word label to query nodes
	for (size_t i = 0; i < n_vertices; i++) {
		labs[i] = matches[i].trainIdx;
		scls[i] = kpts[matches[i].queryIdx].size;
		posx[i] = kpts[matches[i].queryIdx].pt.x;
		posy[i] = kpts[matches[i].queryIdx].pt.y;
	}
	//igraph add labels and degrees attributes 
	igraph_vector_t lab_vec, edge_vec, scale_vec, posx_vec, posy_vec, eweight_vec;
	igraph_vector_view(&eweight_vec, edgeW, igraph_ecount(&mygraph));
	igraph_vector_view(&lab_vec, labels, n_vertices);
	igraph_vector_view(&scale_vec, scales, n_vertices);
	igraph_vector_view(&posx_vec, positionx, n_vertices);
	igraph_vector_view(&posy_vec, positiony, n_vertices);
	
	/*igraph_vector_init(&deg_vec, n_vertices);*/
	//set edge attributes for all edges
	SETEANV(&mygraph, "weight", &eweight_vec);
	SETVANV(&mygraph, "label", &lab_vec);
	SETVANV(&mygraph, "scale", &scale_vec);
	SETVANV(&mygraph, "posx", &posx_vec);
	SETVANV(&mygraph, "posy", &posy_vec);

	igraph_simplify(&mygraph, true, true, 0);
	return true;
}

//if matches is empty, a empty graph is returned
//mygraph must be a uninit graph
bool graph::build(std::vector<DMatch> &matches, std::vector<KeyPoint> &kpts, igraph_t &mygraph) {
	//build from matches result and parameter setting
	if (matches.size() > kpts.size()) {
		std::cout << "graph.buld: matches size cannot larger than the keypoint size!" << std::endl;
		return false;
	}
	graph::igraph_init::attri_init();
	//to ensure the graph building process, for zero matching graphs return empty graph
	if (matches.size() == 0) {
		igraph_empty(&mygraph, 0, IGRAPH_UNDIRECTED);
		SETGAN(&mygraph, "vertices", 0);
		std::cout << "graph.build: warning: zero matches empty graph returned" << std::endl;
		return true;
	}
	clock_t sTime = clock();
	size_t n_vertices = matches.size();
	
	igraph_empty(&mygraph,n_vertices,IGRAPH_UNDIRECTED);
	SETGAS(&mygraph, "name", "kernelGraph");  // set graph name attribute
	SETGAN(&mygraph, "vertices", n_vertices); // set vertices number attribute
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
	//sort the distances and stores the indexes, col represents the sequence!
	for (size_t i = 0; i < n_vertices; i++) {
		std::sort(indexes.col(i).data(), indexes.col(i).data() + n_vertices, [&](size_t left, size_t right) {return (dists.col(i))(left) < (dists.col(i))(right); });
	}
	
	//dynamic C array for labels and scales

	/*igraph_real_t *labels = (igraph_real_t*)malloc(sizeof(igraph_real_t)*n_vertices);
	igraph_real_t* scales = (igraph_real_t*)malloc(sizeof(igraph_real_t) * n_vertices); */// change it to std vector
	std::vector<igraph_real_t> labs(n_vertices), scls(n_vertices),posx(n_vertices),posy(n_vertices);
	igraph_real_t* labels = labs.data();
	igraph_real_t* scales = scls.data();
	igraph_real_t* positionx = posx.data();
	igraph_real_t* positiony = posy.data();
	
	//allocate word label to query nodes
	for (size_t i = 0; i < n_vertices;i++) {
		labs[i] = matches[i].trainIdx;
		scls[i] = kpts[matches[i].queryIdx].size;
		posx[i] = kpts[matches[i].queryIdx].pt.x;
		posy[i] = kpts[matches[i].queryIdx].pt.y;
	}
	//igraph add labels and degrees attributes 
	igraph_vector_t lab_vec, edge_vec,scale_vec,posx_vec,posy_vec;
	
	igraph_vector_view(&lab_vec, labels, n_vertices);
	igraph_vector_view(&scale_vec, scales, n_vertices);
	igraph_vector_view(&posx_vec, positionx, n_vertices);
	igraph_vector_view(&posy_vec, positiony, n_vertices);
	/*igraph_vector_init(&deg_vec, n_vertices);*/

	SETVANV(&mygraph, "label", &lab_vec);
	SETVANV(&mygraph, "scale", &scale_vec);
	SETVANV(&mygraph, "posx", &posx_vec);
	SETVANV(&mygraph, "posy", &posy_vec);
	/*SETVANV(&mygraph, "degree", &deg_vec);*/
	
	//add edge
	std::vector<igraph_real_t> edges;
	edges.reserve(params::maxNumDeg * 2 *n_vertices);
	for (size_t i = 0; i < n_vertices; i++) {
		/* compare the distance and deg limits
		*   a. Here each vertex is limited on the number of their connected vertices
		*	the limiting number  = params::maxNumDeg
		*	b. the edge distance is constrained that edge should not exceed radDegLim * kpts.scale
		*/
		for (size_t j = 1; j<params::maxNumDeg+1 && j < n_vertices; j++) {
			if (dists(i, indexes(j, i)) < params::radDegLim * VAN(&mygraph,"scale",i)) {
				edges.push_back(i);
				edges.push_back(indexes(j,i));
			}
		}
	}
	

	size_t n_edges = edges.size() / 2;
	
	igraph_vector_view(&edge_vec, edges.data(), edges.size());
	igraph_add_edges(&mygraph, &edge_vec, 0);
	SETGAN(&mygraph, "edges", igraph_ecount(&mygraph)); //set edge number attribute

	//set edge weoght attributes, can be deleted later
	std::vector<igraph_real_t> eweight(igraph_ecount(&mygraph), 1);
	igraph_real_t* edgeW = eweight.data();
	igraph_vector_t eweight_vec;
	igraph_vector_view(&eweight_vec, edgeW, igraph_ecount(&mygraph));
	SETEANV(&mygraph, "weight", &eweight_vec);

	igraph_simplify(&mygraph, true, true, 0);
	/*std::cout << " ->graph building spend " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec...." << std::endl;*/
	return true;

}

//source graph will be extended
//RANSAC method to extend the sourceGraph by merging extendGraph
//bestMatches should be the extendGraph as query to the sourceGraph
//best matches should be generated from the compressed descriptors matching!
//new graph stores in the source graph
void graph::extend(igraph_t& sourceGraph, igraph_t& extendGraph, std::vector<DMatch> &bestMatches)
{
	//separate the inliners and outliers
	size_t n_srcVertices = GAN(&sourceGraph, "n_vertices");
	size_t n_exdVertices = GAN(&extendGraph, "n_vertices");

	//extract keypoints from the exdgraph
	igraph_vector_t srcPosx, srcPosy, exdPosx, exdPosy,exdScale,exdLabel;
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
	auto homo = findHomography(exdKpts, srcKpts, mask, cv::RANSAC);

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
			mergeKpts.push_back(KeyPoint(transfKpts[i], VECTOR(exdScale)[i], 0.0, 0, VECTOR(exdLabel)[i]));
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
				continue;
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
				//merge weight
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
	
	
}
