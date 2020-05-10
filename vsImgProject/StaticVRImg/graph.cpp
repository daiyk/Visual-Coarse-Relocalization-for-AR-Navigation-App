#include "graph.h"
#include "fileManager.h"
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <nlohmann/json.hpp>


using namespace cv;
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
	/*fileop::write_graph(graph,"fullgraph","graphml");*/
	
}

bool graph::build(std::vector<DMatch> &matches, std::vector<KeyPoint> &kpts) {
	//build from matches result and parameter setting
	igraph_t mygraph;
	igraph_integer_t n_vertices = matches.size();
	igraph_empty(&mygraph,n_vertices,0);

	//define container for keypoints and compute distance
	Eigen::MatrixXd pos(n_vertices, 2);
	Eigen::MatrixXd dists(n_vertices, n_vertices);
	//loop through and add edges
	for (int i = 0; i < kpts.size();i++) {
		pos.row(i) = Eigen::Vector2d(kpts[i].pt.x, kpts[i].pt.y);
	}

	for (int i = 0; i < n_vertices; i++) {
		//compute the distance norm and store on the matrix
		auto value = (pos.bottomRows(n_vertices - i).rowwise() - pos.row(i)).matrix().rowwise().norm(); // col vector
		dists.row(i).tail(n_vertices - i) = value.transpose();
		dists.col(i).tail(n_vertices - i) = value;
	}

	//sort the distance in ascend order for graph connection
	Eigen::MatrixXi indexes(n_vertices,n_vertices);

	//initialization with continuous numbers
	for (int i = 0; i < n_vertices; i++) {
		int x = 0;
		std::iota(indexes.col(i).data(), indexes.col(i).data()+n_vertices, x++); // sequence assignment for index value
	}

	//sort the distances and stores the indexes, col represents the sequence!
	for (int i = 0; i < n_vertices; i++) {
		std::sort(indexes.col(i).data(), indexes.col(i).data() + n_vertices, [&](int left, int right) {return (dists.col(i))(left) < (dists.col(i))(right); });
	}

	//build graph according to the indexes


	return false;

}