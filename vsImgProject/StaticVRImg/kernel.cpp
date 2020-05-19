#include "kernel.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <set>
#include <unordered_map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
using namespace std;
using namespace Eigen;


kernel::robustKernel::robustKernel(igraph_t &mygraph,int h_max, size_t n_labels):h_max(h_max),n_labels(n_labels) {

}

kernel::robustKernel::~robustKernel() {
    for (auto& it : graphs) {
        igraph_destroy(&it);
    }
}

void kernel::robustKernel::push_back(igraph_t& newgraph) {
    this->graphs.push_back(newgraph);
}

void kernel::robustKernel::graphPrepro(igraph_t& graph) {
    igraph_vector_t edges, labels;
    igraph_vector_init(&edges, 0);
    igraph_vector_init(&labels, 0);
    igraph_get_edgelist(&graph, &edges, true);

    //get the label vecyor
    VANV(&graph, "label", &labels);

    size_t edgeLen = igraph_vector_size(&edges);
    /*MatrixXd edgesMat(edgeLen / 2, 2);

    for (size_t i = 0; i < edgeLen / 2; i++)
    {
        edgesMat(i, 0) = VECTOR(edges)[i];
        edgesMat(i, 1) = VECTOR(edges)[i + edgeLen / 2];
    }*/

    //build inverted indices
    unordered_map<int, vector<size_t>> inverted_index;
    for (size_t i = 0; i < igraph_vector_size(&labels); i++) {
        inverted_index[VECTOR(labels)[i]].push_back(i);
    }
    this->inverted_indices.push_back(inverted_index);
    //stores the number of vertices and edges
    this->ver_nums.push_back(igraph_vector_size(&labels));
    this->edge_nums.push_back(edgeLen / 2);
    
    //compute normalized edge norm for each graph
    //??

    igraph_vector_destroy(&edges);
    igraph_vector_destroy(&labels);
}

double kernel::robustKernel::robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, int i, int j) {

    std::vector<double>kernelVals;
    //get the adjacency list
    //igraph_adjlist_t adjlist_i, adjlist_j;
    //igraph_adjlist_init(&graph_i, &adjlist_i, IGRAPH_ALL);
    //igraph_adjlist_init(&graph_j, &adjlist_j, IGRAPH_ALL);

    igraph_t& graph_i = this->graphs[i];
    igraph_t& graph_j = this->graphs[j];
    //compute the edge norm
    double edge_norm1 = 1.0 / this->edge_nums[i];
    double edge_norm2 = 1.0 / this->edge_nums[j];
    //iterate through the two sets of vertices and compute the kernel values
    igraph_vector_t nei1_lab,nei2_lab;
    igraph_vector_init(&nei1_lab, 0);
    igraph_vector_init(&nei2_lab, 0);
    for (size_t i = 0; i < vert1.size(); i++) {
        igraph_vs_t vert1_nei;
        igraph_vs_adj(&vert1_nei, vert1[i], IGRAPH_ALL);
        igraph_cattribute_VANV(&graph_i, "label", vert1_nei, &nei1_lab);

        //create the nei vector
        igraph_vector_t nei1_vec;
        igraph_vector_init(&nei1_vec, this->n_labels);
        for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
            VECTOR(nei1_vec)[VECTOR(nei1_lab)[m]] += edge_norm1;
        }
        for (size_t j = 0; j < vert2.size(); j++) {
            igraph_vs_t vert2_nei;
            igraph_cattribute_VANV(&graph_j, "label", vert2_nei, &nei2_lab);

            //create the nei vector
            igraph_vector_t nei2_vec;
            igraph_vector_init(&nei2_vec, this->n_labels);
            for (size_t n = 0; n < igraph_vector_size(&nei2_lab); n++) {
                VECTOR(nei2_vec)[VECTOR(nei2_lab)[n]] += edge_norm2;
            }

            //multiple two vector and get the kernel value
            igraph_vector_mul(&nei2_vec, &nei1_vec);
            kernelVals.push_back(igraph_vector_sum(&nei2_vec));

            igraph_vector_destroy(&nei2_vec);
            igraph_vs_destroy(&vert2_nei);
        }
        igraph_vector_destroy(&nei1_vec);
        igraph_vs_destroy(&vert1_nei);

    }

    igraph_vector_destroy(&nei1_lab);
    igraph_vector_destroy(&nei2_lab);

    return *max_element(kernelVals.begin(), kernelVals.end());
}
    

igraph_matrix_t kernel::robustKernel::robustKernelCom() {
    //count label value and construct index corresponding for adjancy matrix
    std::set<int> label_sets;
    int n_graphs = this->graphs.size();
    //iterate through whole sets of graphs. and insert each graph's inverted_tree to count for the unique labels
    for (int i = 0; i < this->inverted_indices.size(); i++) {
        for (const auto &inv_idx : this->inverted_indices[i]) {
            label_sets.insert(inv_idx.first);
        }
    }
    //total unique labels num
    int n_labels = label_sets.size();

    //build label indices
    std::map<int, int> label_index;
    int idx = 0;
    //mark each label for their positions at the 
    for (auto i : label_sets) {
        label_index.insert(pair<int, int>(i, idx));
        idx++;
    }

    //Compute edge norm for each graph
    //??

    //build neighborhood label vector

    //iterate the predefined iteration and compute the kernel values
    igraph_matrix_t k_matrix;
    igraph_matrix_init(&k_matrix, n_graphs, n_graphs);
    igraph_matrix_null(&k_matrix);


    for (int h = 0; h < h_max; h++) {
        //iterate over all stored graphs
        for (int i = 0; i < n_graphs; i++) {
            for (int j = 0; j < n_graphs; j++) {
                auto& inv1 = this->inverted_indices[i];
                auto& inv2 = this->inverted_indices[j];

                for (auto& val : label_sets) {
                    if (inv1.count(val) && inv2.count(val)) {
                        auto& vers1 = inv1[val];
                        auto& vers2 = inv2[val];
                        double kernelval = robustKernelVal(vers1, vers2, i, j);
                        if (i == j) {
                            kernelval *= 0.5;
                        }
                        MATRIX(k_matrix, i, j) = MATRIX(k_matrix, i, j) + kernelval;
                        MATRIX(k_matrix, j, i) = MATRIX(k_matrix, j, i) + kernelval;
                    }
                }
            }
        }
    }

    //get the total sum of neighboring comparison value to itself
    double sum_self_kernel = 0;
    for (int i = 0; i < n_graphs; i++) {
        sum_self_kernel += MATRIX(k_matrix, i, i);
    }
    igraph_matrix_scale(&k_matrix, sum_self_kernel);
    return k_matrix;
}


double kernel::robustKernel::kernelValue(const vector<int>& map1, const vector<int>& map2, int& i, int& j, vector<int>& num_v, MatrixXd& node_nei)
{
    //calculate the kernel value increments
    vector<double> values;
    int start1 = 0, start2 = 0;
    for (int m = 0; m < i; m++) {
        start1 += num_v[m];
    } //labels the start of ith graph
    for (int m = 0; m < j; m++) {
        start2 += num_v[m];
    } //labels the start of jth graph


    for (auto v1 : map1) {
        for (auto v2 : map2) {
            VectorXd label_vec1 = node_nei.row(start1 + v1);
            VectorXd label_vec2 = node_nei.row(start2 + v2);

            //compute kernel value
            values.push_back(label_vec1.dot(label_vec2));
        }

    }

    //return the largest dot product value
    return *max_element(values.begin(), values.end());
}
/**
 *E: the vector of edge matrix, edge = (E.ele[,0], E.ele[,1]), edge strength=E.ele[,2]
 *V_label: label values for per-graph per node, outer length V_label must equal E outer length
 *h_max: the total steps or iterations
 *V_label: label value(int) for each graph
 *num_v: num of vertices for each graph
 *num_e: num of edge for each graph
 *K_mat: kernel matrix that stores value for graphs comparison.

**/
void kernel::robustKernel::wlRobustKernel(vector<MatrixXi>& E, vector<vector<int>>& V_label, vector<int>& num_v, vector<int>& num_e, int h_max, MatrixXd& K_mat)
{
    //E is the vector of edge matrices, but only two graphs
    int n = 2;                                             //num of graphs !change to E.size for multigraph processing
    int v_all = accumulate(num_v.begin(), num_v.end(), 0); // compute the num of vertices
    // int e_all = *max_element(num_e.begin(), num_e.end()); //find the max number of 

    //insert element to the set and count the total unique values
    set<int> n_label;

    //map the label value to the index in label vector
    std::map<int, int> label_index;

    //build vector that stores the edge-normallized weight for graphs 
    vector<VectorXd> norm_edge(n);

    //build inverted index for label value
    //vertex num system use per-graph based
    //inverted index means label value as index and stores the vertex ind that belongs to this label
    vector<unordered_map<int, vector<int> > > inverted_index;

    for (int i = 0; i < V_label.size(); i++)
    {
        unordered_map<int, vector<int>> graph_inverted_index;
        for (int j = 0; j < V_label[i].size(); j++)
        {
            //use label value as index
            (graph_inverted_index[V_label[i][j]]).push_back(j);
            n_label.insert(V_label[i][j]);
        }
        inverted_index.push_back(graph_inverted_index);
    }
    //count the num of labels
    int n_label_vals = n_label.size();

    //insert according to the labels vector's index ordered set
    //use to mark the idx of each label value in the neighborhood label vector (see following) 
    int idx = 0;
    for (auto it : n_label)
    {
        label_index.insert(pair<int, int>(it, idx)); //sorted label values
        idx++;
    }

    //weight the edge strength graph-wise
    for (size_t i = 0; i < n; i++)
    {
        double edge_w = 0;
        // sum of edge weights
        edge_w = E[i].col(2).sum();
        norm_edge[i].resize(num_v[i]);

        norm_edge[i] = E[i].col(2).cast<double>();
        norm_edge[i].array() /= edge_w;
    }

    //build neighborhood label vector for each vertex
    MatrixXd node_nei_vec = MatrixXd::Zero(v_all, n_label_vals);
    int raise = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < E[i].rows(); j++)
        {
            int edge_0_label = V_label[i][E[i](j, 0)];
            int edge_1_label = V_label[i][E[i](j, 1)];

            //add to the neighborhood list
            //need to consider the offset for different graph
            //label_index returns the label value's index
            double val1 = norm_edge[i](j);
            int val2 = E[i](j, 1);
            int val3 = E[i](j, 0);
            int val4 = label_index[edge_0_label];
            node_nei_vec(E[i](j, 1) + raise, label_index[edge_0_label]) += norm_edge[i](j);
            node_nei_vec(E[i](j, 0) + raise, label_index[edge_1_label]) += norm_edge[i](j);
        }
        raise += num_v[i];
    }

    //iterate the predefined iteration and update kernel value
    K_mat.resize(n, n);
    K_mat.setZero();
    for (int h = 0; h < h_max; h++)
    {
        //iterate all graphs
        for (int i = 0; i < n; i++)
        {
            int offset = 0;
            for (int j = i; j < n; j++)
            {
                auto& inv_vet1 = inverted_index[i];
                auto& inv_vet2 = inverted_index[j];
                //TODO iterate over all label value vertex! change the vertex numbering way!
                for (auto& val : n_label)
                {
                    //only increment the kernel value when same label is found in both graphs
                    if (inv_vet1.count(val) && inv_vet2.count(val)) {
                        auto& map1 = inv_vet1[val];
                        auto& map2 = inv_vet2[val];
                        double kernelInc = kernelValue(map1, map2, i, j, num_v, node_nei_vec);
                        K_mat(i, j) += kernelInc;
                        K_mat(j, i) += kernelInc;
                    }
                }
            }
        }
    }
    //the kernel value of each graph to itself, k_mat(i,i) is counted twice, thus need to get back
    K_mat.diagonal() /= 2;

    //normalize the kernel value matrix
    double sum_kernel = K_mat.diagonal().sum();

    //divide the kernel matrix with the total sum of self-kernel
    K_mat /= sum_kernel;

    //

    //still need to divide the graph's comparison score by the total sum of neighborhood comparisons of itself.
}
