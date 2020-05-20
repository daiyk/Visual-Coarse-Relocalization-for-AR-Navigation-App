#include "graph.h"
#include "fileManager.h"
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <nlohmann/json.hpp>


using namespace cv;
using params = fileManager::parameters;
void graph::graphTest() {
	igraph_integer_t diameter; igraph_t graph;
	igraph_rng_seed(igraph_rng_default(), 42);
	igraph_erdos_renyi_game(&graph, IGRAPH_ERDOS_RENYI_GNP, 1000, 5.0 / 1000, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
	igraph_diameter(&graph, &diameter, 0, 0, 0, IGRAPH_UNDIRECTED, 1); 
	printf("Diameter of a random graph with average degree 5: %d\n", (int)diameter);
	igraph_destroy(&graph);
}

//build full graph and return it
bool graph::buildFull(igraph_t &graph, int n, bool directed) {
	igraph_integer_t num = n;
	igraph_bool_t loops = false;

	int status = igraph_full(&graph, n, directed, loops);

	if (status == IGRAPH_EINVAL) {
		std::cout << "build full graph: Invalid number of graph vertices" << std::endl;
		return false;
	}
	
}

bool graph::build(std::vector<DMatch> &matches, std::vector<KeyPoint> &kpts, igraph_t &mygraph) {
	//build from matches result and parameter setting
	if (matches.size() != kpts.size()) {
		std::cout << "graph.buld: matches and kpts size doesn't match!" << std::endl;
		return false;
	}
	clock_t sTime = clock();
	size_t n_vertices = matches.size();
	igraph_i_set_attribute_table(&igraph_cattribute_table);
	igraph_empty(&mygraph,n_vertices,IGRAPH_UNDIRECTED);
	SETGAS(&mygraph, "name", "kernelGraph");

	//define container for keypoints and compute distance
	Eigen::MatrixXd pos(n_vertices, 2);
	Eigen::MatrixXd dists(n_vertices, n_vertices);
	//loop through and add edges
	for (size_t i = 0; i < kpts.size();i++) {
		pos.row(i) = Eigen::Vector2d(kpts[i].pt.x, kpts[i].pt.y);
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
		std::iota(indexes.col(i).data(), indexes.col(i).data()+n_vertices, x++); // sequence assignment for index value
	}
	//sort the distances and stores the indexes, col represents the sequence!
	for (size_t i = 0; i < n_vertices; i++) {
		std::sort(indexes.col(i).data(), indexes.col(i).data() + n_vertices, [&](size_t left, size_t right) {return (dists.col(i))(left) < (dists.col(i))(right); });
	}
	
	//dynamic C array for labels and scales

	/*igraph_real_t *labels = (igraph_real_t*)malloc(sizeof(igraph_real_t)*n_vertices);
	igraph_real_t* scales = (igraph_real_t*)malloc(sizeof(igraph_real_t) * n_vertices); */// change it to std vector
	std::vector<igraph_real_t> labs(n_vertices), scls(n_vertices);
	igraph_real_t* labels = labs.data();
	igraph_real_t* scales = scls.data();
	//allocate word label to query nodes
	for (size_t i = 0; i < n_vertices;i++) {
		labs[matches[i].queryIdx] = matches[i].trainIdx;
		scls[matches[i].queryIdx] = kpts[matches[i].queryIdx].size;
	}
	//igraph add labels and degrees attributes 
	igraph_vector_t lab_vec, edge_vec,scale_vec;
	igraph_vector_view(&lab_vec, labels, n_vertices);
	igraph_vector_view(&scale_vec, scales, n_vertices);
	/*igraph_vector_init(&deg_vec, n_vertices);*/

	SETVANV(&mygraph, "label", &lab_vec);
	SETVANV(&mygraph, "scale", &scale_vec);
	/*SETVANV(&mygraph, "degree", &deg_vec);*/
	
	//add edge
	std::vector<igraph_real_t> edges;
	edges.reserve(params::maxNumDeg * 2 *n_vertices);
	for (size_t i = 0; i < n_vertices; i++) {
		//compare the distance and deg limits
		for (size_t j = 1; j<params::maxNumDeg+1 && j < n_vertices; j++) {
			if (dists(i, indexes(j, i)) < params::radDegLim * VAN(&mygraph,"scale",i)) {
				edges.push_back(i);
				edges.push_back(indexes(j,i));
			}
		}
	}
	igraph_vector_view(&edge_vec, edges.data(), edges.size());
	igraph_add_edges(&mygraph, &edge_vec, 0);

	igraph_simplify(&mygraph, true, true, 0);
	free(labels);
	free(scales);
	std::cout << " ->graph building spend " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " sec...." << std::endl;
	return true;

}