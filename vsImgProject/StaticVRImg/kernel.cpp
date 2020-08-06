#include "kernel.h"
#include "helper.h"
#include "fileManager.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <set>
#include <unordered_map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <igraph.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
using namespace std;
using namespace Eigen;
using params = fileManager::parameters;

/****************************/
/****************************/
/***** CLASS ROBUSTKERNEL ******/
/****************************/
/****************************/


kernel::robustKernel::robustKernel(int h_max, size_t n_labels):h_max(h_max),n_labels(n_labels) {

}

kernel::robustKernel::~robustKernel() {
    //destory every graphs in the kernel container
    for (auto& i : this->graphs) {
        igraph_destroy(&i);
    }
}

void kernel::robustKernel::push_back(igraph_t newgraph) {
    igraph_t newpushgraph;
    igraph_copy(&newpushgraph, &newgraph);
    this->graphs.push_back(newpushgraph);
    this->graphPrepro(newpushgraph);
}

void kernel::robustKernel::graphPrepro(igraph_t& graph) {
    igraph_vector_t edges, labels;
    igraph_vector_init(&edges, 0);
    igraph_vector_init(&labels, 0);
    igraph_get_edgelist(&graph, &edges, true);

    //get the label vecyor
    VANV(&graph, "label", &labels);

    size_t edgeLen = igraph_vector_size(&edges);

    //build inverted indices
    unordered_map<int, vector<size_t>> inverted_index;
    for (size_t i = 0; i < igraph_vector_size(&labels); i++) {
        inverted_index[VECTOR(labels)[i]].push_back(i);
    }

    
     //insert to label_sets for the calculation of total unique labels
    double kernelval = 0.0;
    for (auto& inv_idx : inverted_index) {
        this->label_sets.insert(inv_idx.first);
        //compute raw selfkernel value for each query graphs
        kernelval += robustKernelVal(inv_idx.second, inv_idx.second, graph, graph);
    }
    this->raw_self_kernel.push_back(kernelval);
     
    this->inverted_indices.push_back(inverted_index);
    //stores the number of vertices and edges
    this->ver_nums.push_back(igraph_vector_size(&labels));
    this->edge_nums.push_back(edgeLen / 2);
    
    igraph_vector_destroy(&edges);
    igraph_vector_destroy(&labels);
}

double kernel::robustKernel::robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, igraph_t& graph_i, igraph_t& graph_j) {

    std::vector<double> kernelVals;
    //get the adjacency list
    //igraph_adjlist_t adjlist_i, adjlist_j;
    //igraph_adjlist_init(&graph_i, &adjlist_i, IGRAPH_ALL);
    //igraph_adjlist_init(&graph_j, &adjlist_j, IGRAPH_ALL);

    //igraph_t& graph_i = this->graphs[i];
    //igraph_t& graph_j = this->graphs[j];
    //compute the edge norm

    //?? edge norm should be obatained from edge weight computation
    
    //iterate through the two sets of vertices and compute the kernel values 
    for (size_t i = 0; i < vert1.size(); i++) {
        igraph_vector_t nei1_lab;
        igraph_vector_init(&nei1_lab, 0);
        igraph_vs_t vert1_nei;
        igraph_vs_adj(&vert1_nei, vert1[i], IGRAPH_ALL);
        igraph_cattribute_VANV(&graph_i, "label", vert1_nei, &nei1_lab);

        if (igraph_vector_size(&nei1_lab) == 0) { 
            igraph_vector_destroy(&nei1_lab);
            continue; 
        }
        //create the nei vector
        std::vector<int> nei1_vec(this->n_labels, 0);
        //igraph_vector_t nei1_vec;
        //igraph_vector_init(&nei1_vec, this->n_labels);
        for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
            nei1_vec[(int)VECTOR(nei1_lab)[m]] += 1;
        }
        for (size_t j = 0; j < vert2.size(); j++) {
            igraph_vector_t nei2_lab;
            igraph_vector_init(&nei2_lab, 0);
            igraph_vs_t vert2_nei;
            igraph_vs_adj(&vert2_nei, vert2[j], IGRAPH_ALL);
            igraph_cattribute_VANV(&graph_j, "label", vert2_nei, &nei2_lab);

            if (igraph_vector_size(&nei2_lab) == 0) { 
                igraph_vector_destroy(&nei2_lab); 
                continue; 
            }
            //create the nei_vector
            std::vector<int> nei2_vec(this->n_labels, 0);
            //igraph_vector_t nei2_vec;
            //igraph_vector_init(&nei2_vec, this->n_labels);
            for (size_t n = 0; n < igraph_vector_size(&nei2_lab); n++) {
                nei2_vec[(int)VECTOR(nei2_lab)[n]] += 1;
            }

            //rebuild the vector with element-wise min
            std::vector<int> cwiseMinVec;
            cwiseMinVec.reserve(this->n_labels);
            std::transform(nei1_vec.begin(), nei1_vec.end(), nei2_vec.begin(), std::back_inserter(cwiseMinVec), [](double a, double b) {return std::min(a, b); });


            //multiple two vector and get the kernel value
            //igraph_vector_mul(&nei2_vec, &nei1_vec);
            //kernelVals.push_back(igraph_vector_sum(&nei2_vec));
            kernelVals.push_back(std::inner_product(cwiseMinVec.begin(), cwiseMinVec.end(), cwiseMinVec.begin(), 0.0));

            igraph_vector_destroy(&nei2_lab);
            //igraph_vector_destroy(&nei2_vec);
            igraph_vs_destroy(&vert2_nei);
        }
        igraph_vector_destroy(&nei1_lab);
        //igraph_vector_destroy(&nei1_vec);
        igraph_vs_destroy(&vert1_nei);

    }

    //debug
    if (!kernelVals.empty()) {
        return *max_element(kernelVals.begin(), kernelVals.end());
    }
    else
        return 0.0;
}

//1th arg: the database graphs that query graphs need to search among database
std::vector < std::vector<double>> kernel::robustKernel::robustKernelCompWithQueryArray(std::vector<igraph_t>& database_graphs) {
    //count label value and construct index corresponding for adjancy matrix
    /*std::set<int> label_sets;*/
    int n_query_graphs = this->graphs.size();
    int n_database_graphs = database_graphs.size();
    auto verNums = this->ver_nums;
    auto edgeNums = this->edge_nums;
    auto invertedIndices = this->inverted_indices;
    auto labelSets = this->label_sets;
    auto rawSelfKernel = this->raw_self_kernel;

    //iterate through whole sets of graphs. and insert each graph's inverted_tree to count for the unique labels
    /*for (int i = 0; i < this->inverted_indices.size(); i++) {
        for (const auto& inv_idx : this->inverted_indices[i]) {
            label_sets.insert(inv_idx.first);
        }
    }*/
    //total unique labels num

    //preprocessing database graphs
    for (int i = 0; i < database_graphs.size(); i++) {
        igraph_vector_t edges, labels;
        igraph_vector_init(&edges, 0);
        igraph_vector_init(&labels, 0);
        igraph_get_edgelist(&database_graphs[i], &edges, true);

        //get the label vector
        VANV(&database_graphs[i], "label", &labels);

        size_t edgeLen = igraph_vector_size(&edges);

        //build inverted indices
        unordered_map<int, vector<size_t>> inverted_index;
        for (size_t i = 0; i < igraph_vector_size(&labels); i++) {
            inverted_index[VECTOR(labels)[i]].push_back(i);
        }

        //insert to label_sets for the calculation of total unique labels
        //+ compute the self-kernel values
        double kernelval = 0.0;
        for (auto inv_idx : inverted_index) {
            labelSets.insert(inv_idx.first);
            kernelval += robustKernelVal(inv_idx.second, inv_idx.second, database_graphs[i], database_graphs[i]);
        }
        rawSelfKernel.push_back(kernelval);
        invertedIndices.push_back(inverted_index);
        //stores the number of vertices and edges
        verNums.push_back(igraph_vector_size(&labels));
        edgeNums.push_back(edgeLen / 2);

        //compute normalized edge norm for each graph
        igraph_vector_destroy(&edges);
        igraph_vector_destroy(&labels);
    }

    int n_labels = labelSets.size();
    std::vector<std::vector<double>> k_matrix(n_query_graphs, std::vector<double>(n_database_graphs, 0.0));
    raw_scores = std::vector<std::vector<int>>(n_query_graphs, std::vector<int>(n_database_graphs, 0));
    /*igraph_matrix_t k_matrix;
    igraph_matrix_init(&k_matrix, n_query_graphs, n_database_graphs);
    igraph_matrix_null(&k_matrix);*/
    omp_set_num_threads(6);
    //iterate to compute each query image to database image values
    for (int h = 0; h < h_max; h++) {
        //iterate over all stored graphs
#pragma omp parallel
        {
#pragma omp for schedule(dynamic)
            for (int i = 0; i < n_query_graphs; i++) {
                //start from i to avoid multicount the values?? but set to start from 0 makes the computation happens twice for (i,j) and (j,i)
                auto& inv1 = invertedIndices[i]; //query graph
                for (int j = n_query_graphs; j < n_query_graphs + n_database_graphs; j++) { //j=0 if computed from division by sum of two self-kernel values
                    double edge_norm = 0.0;
                    auto& inv2 = invertedIndices[j]; //database graph
                    if (edgeNums[i] + edgeNums[j] != 0) {
                        edge_norm = 1.0 / (edgeNums[i] + edgeNums[j]);
                    }
                    else {
                        edge_norm = 0.0;
                    }
                    for (auto& val : labelSets) {
                        if (inv1.count(val) && inv2.count(val)) {
                            auto& vers1 = inv1[val];
                            auto& vers2 = inv2[val];
                            double kernelval = robustKernelVal(vers1, vers2, this->graphs[i], database_graphs[j - n_query_graphs]);
                            //raw_scores is only for debug
                            raw_scores[i][j - n_query_graphs] = raw_scores[i][j - n_query_graphs] + kernelval;
                            kernelval *= edge_norm * edge_norm;
                            /*auto ptr = igraph_matrix_e_ptr(&k_matrix, 1, 0);
                            *ptr = *ptr+kernelval;
                            MATRIX(k_matrix, i, j-n_query_graphs) = MATRIX(k_matrix, i, (j-n_query_graphs)) + kernelval;*/
                            k_matrix[i][j - n_query_graphs] = k_matrix[i][j - n_query_graphs] + kernelval;
                        }
                    }
                }
            }
        }
    }
    //get the total sum of neighboring comparison value to itself
       //print the matrix value and check out
       //original paper  

       /*double sum_self_kernel = 0.0;
       for (int i = 0; i < n_graphs; i++) {
           sum_self_kernel += MATRIX(k_matrix, i, i);
       }
       igraph_matrix_scale(&k_matrix, 2.0/sum_self_kernel);
       return k_matrix;*/

       //use alternative equation for kernel value comparison
       //for (int i = 0; i < n_graphs; i++) {
       //    for (int j = 0; j < n_graphs; j++) {
       //        std::cout << MATRIX(k_matrix, i, j) << "\t";
       //    }
       //    std::cout << std::endl;
       //}
       //double sum_self_kernel = 1.0;
       //for (int i = 0; i < n_graphs; i++) {
       //    sum_self_kernel *= MATRIX(k_matrix, i, i);
       //}
       ////loop all element and compute the score
       //igraph_matrix_scale(&k_matrix, 1.0/sqrt(sum_self_kernel));
       //return k_matrix;
       //use stored self-kernel value to 

       //divide by the selfkernel value for probability calculation
       //for(int i=0;i<n_query_graphs;i++)
       //    for (int j = 0; j < n_database_graphs; j++) {
       //        if (edgeNums[i] + edgeNums[j + n_query_graphs] != 0) {
       //            double edge_norm = 0.0;
       //            edge_norm = 1.0 / (edgeNums[i] + edgeNums[j + n_query_graphs]);
       //            /*MATRIX(k_matrix, i, j) = MATRIX(k_matrix, i, j) / (this->raw_self_kernel[i] * edge_norm * edge_norm);*/
       //            k_matrix[i][j] = k_matrix[i][j] / (this->raw_self_kernel[i] * edge_norm * edge_norm);
       //        }   
       //    }
    helper::computeScore1(k_matrix, edgeNums, rawSelfKernel);
    return k_matrix;
}

   


//compute the robust kernel values for every pair in graph vector: default w.r.t [0] graph as query graph
igraph_matrix_t kernel::robustKernel::robustKernelCom() {
    //count label value and construct index corresponding for adjancy matrix
    int n_graphs = this->graphs.size();
    ////iterate through whole sets of graphs. and insert each graph's inverted_tree to count for the unique labels
    //for (int i = 0; i < this->inverted_indices.size(); i++) {
    //    for (const auto &inv_idx : this->inverted_indices[i]) {
    //        label_sets.insert(inv_idx.first);
    //    }
    //}
    //total unique labels num
    int n_labels = this->label_sets.size();

    //build label indices
    //std::map<int, int> label_index;
    //int idx = 0;
    ////mark each label for their positions at the label_set, and build invert index
    //for (auto i : label_sets) {
    //    label_index.insert(pair<int, int>(i, idx));
    //    idx++;
    //}

    //Compute edge norm for each graph
    //??

    //build neighborhood label vector

    //iterate the predefined iteration and compute the kernel values
    igraph_matrix_t k_matrix;
    igraph_matrix_init(&k_matrix, n_graphs, n_graphs);
    igraph_matrix_null(&k_matrix);

    double edge_num_sum=0.0, edge_norm;
    for (int i = 0; i < n_graphs; i++) {
        edge_num_sum += this->edge_nums[i];
    }
    
    
    if (!helper::isEqual(edge_num_sum, 0.0)) {
        edge_norm = 1.0 / edge_num_sum;
    }
    else {
        edge_norm = 0.0;
    }
    for (int h = 0; h < h_max; h++) {
        //iterate over all stored graphs
        for (int i = 0; i < n_graphs; i++) {
            //start from i to avoid multicount the values?? but set to start from 0 makes the computation happens twice for (i,j) and (j,i)
            for (int j = i; j < n_graphs; j++) { //j=0 if computed from division by sum of two self-kernel values
                auto& inv1 = this->inverted_indices[i];
                auto& inv2 = this->inverted_indices[j];
                
                for (auto& val : this->label_sets) {
                    if (inv1.count(val) && inv2.count(val)) {
                        auto& vers1 = inv1[val];
                        auto& vers2 = inv2[val];
                        double kernelval = robustKernelVal(vers1, vers2, this->graphs[i], this->graphs[j]);
                        kernelval *= edge_norm * edge_norm;
                        if (i == j) { 
                            kernelval *= 0.5;
                        }
                        //add value to (i,j) and (j,i), this is the reason why we need to 0.5*kernelval for i=j
                        MATRIX(k_matrix, i, j) = MATRIX(k_matrix, i, j) + kernelval;
                        MATRIX(k_matrix, j, i) = MATRIX(k_matrix, j, i) + kernelval;
                    }
                }
            }
        }
    }

    //get the total sum of neighboring comparison value to itself
    //print the matrix value and check out
    //original paper  

    /*double sum_self_kernel = 0.0;
    for (int i = 0; i < n_graphs; i++) {
        sum_self_kernel += MATRIX(k_matrix, i, i);
    }
    igraph_matrix_scale(&k_matrix, 2.0/sum_self_kernel);
    return k_matrix;*/

    //use alternative equation for kernel value comparison
    //for (int i = 0; i < n_graphs; i++) {
    //    for (int j = 0; j < n_graphs; j++) {
    //        std::cout << MATRIX(k_matrix, i, j) << "\t";
    //    }
    //    std::cout << std::endl;
    //}
    //double sum_self_kernel = 1.0;
    //for (int i = 0; i < n_graphs; i++) {
    //    sum_self_kernel *= MATRIX(k_matrix, i, i);
    //}
    ////loop all element and compute the score
    //igraph_matrix_scale(&k_matrix, 1.0/sqrt(sum_self_kernel));
    //return k_matrix;

    /*for (int i = 0; i < n_graphs; i++) {
        for (int j = 0; j < n_graphs; j++) {
            std::cout << MATRIX(k_matrix, i, j) << "\t";
        }
        std::cout << std::endl;
    }*/

    //use the query graph as normalization
    //!DEFAULT (0,0) is the query graph!


    //use the raw_self_kernel value instead
    
    double sum_self_kernel = pow(MATRIX(k_matrix,0,0),2);
    double raw_self_kernel = pow(this->raw_self_kernel[0]*edge_norm*edge_norm,2);
    //loop all element and compute the 
    if (!helper::isEqual(raw_self_kernel, 0.0)) {
        igraph_matrix_scale(&k_matrix, 1.0 / sqrt(raw_self_kernel));
    }
    return k_matrix;
}

/****************************/
/****************************/
/***** CLASS COVISMAP ******/
/****************************/
/****************************/
/*******first the graph class build fully-connected graph and push it to the vector covisMap then build inverted index and cliques from the map******/
void kernel::covisMap::process(igraph_t& mygraph) {
    //process the graph, build covismap
    igraph_vector_t labs;
    igraph_vector_init(&labs, 0);
    VANV(&mygraph, "label", &labs);
    //special for the first graph
    int ncols = igraph_sparsemat_ncol(&this->map);
    std::cout << "number of cols: " << ncols << std::endl;
    std::vector<bool> unique_labs(this->kCenters,true);
    //mark label to the corresponding position
    //here we ensure the unique of labels, or we only records whether a label is shown but not its frequency
    for (int i = 0; i < igraph_vector_size(&labs); i++) {
        //triplet value and we don't force label value = 1 but rather record the number of same label value nodes
        if (unique_labs[VECTOR(labs)[i]]) {
            igraph_sparsemat_entry(&this->map, VECTOR(labs)[i], ncols - 1, 1);
            //record obs to the inverted index tree
            this->inverted_tree[VECTOR(labs)[i]].push_back(ncols - 1);
            unique_labs[VECTOR(labs)[i]] = false;
        }
    }
    igraph_vector_destroy(&labs);
    igraph_sparsemat_add_cols(&this->map, 1);
}

//retrieve by landmark's labels
std::vector<std::vector<int>> kernel::covisMap::retrieve(igraph_t& queryGraph) {
    //extract labels from queryGraph and search among the inverted tree and labels
    igraph_vector_t labs;
    igraph_vector_init(&labs, 0);
    VANV(&queryGraph, "label", &labs);

    //iterate through the labs and retrieve the corresponding graphs/cliques
    std::vector<bool> unique_labs(this->kCenters,true);
    std::map<int, int> retrieve_stat;
    for (int i = 0; i < igraph_vector_size(&labs); i++) {
        if (unique_labs[VECTOR(labs)[i]]) {
            auto graph_vec = this->inverted_tree[VECTOR(labs)[i]];

            //iterate the graph idx and add to the statist
            for (auto j : graph_vec) {
                retrieve_stat[j] = retrieve_stat[j] + 1;
            }
            unique_labs[VECTOR(labs)[i]] = false;
        }
    }
    //check for minimum condidate condition
    std::vector<int> selected_graphs;
    selected_graphs.reserve(retrieve_stat.size());
    for (auto it : retrieve_stat) {
        if ((double)it.second / retrieve_stat.size() > params::PCommonwords) {
            selected_graphs.push_back(it.first);
        }
    }
    //extract the selected graphs from cliques
    igraph_sparsemat_t M, M_, M_tM_, M_t;
    igraph_vector_int_t q;
    igraph_vector_int_view(&q, selected_graphs.data(), selected_graphs.size());
    igraph_sparsemat_compress(&this->map, &M);
    igraph_sparsemat_dupl(&M);
    igraph_sparsemat_index(&M, nullptr, &q, &M_, nullptr);

    //self matrix multiplication to filter cliques extension
    igraph_sparsemat_transpose(&M_,&M_t,1);
    igraph_sparsemat_multiply(&M_t, &M_, &M_tM_);
    igraph_matrix_t M_cols_sum,M_tM_matrix;
    igraph_matrix_init(&M_cols_sum, 0, 0);
    igraph_matrix_init(&M_tM_matrix, 0, 0);
    igraph_vector_t cols_sum;
    igraph_vector_init(&cols_sum, 0);
    igraph_sparsemat_as_matrix(&M_cols_sum,&M_);
    igraph_sparsemat_as_matrix(&M_tM_matrix, &M_tM_);
    igraph_matrix_colsum(&M_cols_sum,&cols_sum);

    //check the covisibility params and filter the graphs
    //check all (i,i+1) to see if it meets the requirement
    std::vector<std::vector<int>> cliques;
    std::vector<int> single_clique;
    for (int i = 0; i < igraph_vector_size(&cols_sum) - 1; i++) {
        single_clique.push_back(selected_graphs[i]);
        if (double(MATRIX(M_tM_matrix, i, i + 1) / VECTOR(cols_sum)[i])>params::PCliques) {
            if (i == igraph_vector_size(&cols_sum) - 2)
            {
                single_clique.push_back(selected_graphs[i+1]);
                cliques.push_back(single_clique);
            }
            continue;
        }
        else {
            cliques.push_back(single_clique);
            single_clique.clear();
        }
    }
    igraph_vector_destroy(&labs);
    igraph_vector_destroy(&cols_sum);
    igraph_sparsemat_destroy(&M);
    igraph_sparsemat_destroy(&M_tM_);
    igraph_sparsemat_destroy(&M_t);
    igraph_matrix_destroy(&M_cols_sum);
    igraph_matrix_destroy(&M_tM_matrix);
    return cliques;
}

void kernel::covisMap::printMap() {
    igraph_matrix_t res;
    igraph_sparsemat_t compressed;
    igraph_sparsemat_compress(&this->map, &compressed);
    igraph_matrix_init(&res, 0, 0);
    igraph_sparsemat_as_matrix(&res, &compressed);
    for (int i = 0; i < igraph_matrix_nrow(&res); i++) {
        for (int j = 0; j < igraph_matrix_ncol(&res); j++) {
            std::cout << MATRIX(res, i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void kernel::covisMap::processSampleLoc() {

}
//double kernel::robustKernel::robustKernelValTest(std::vector<size_t>& vert1, std::vector<size_t>& vert2, int i, int j, double edge_norm) {
//
//    std::vector<double>kernelVals;
//    //get the adjacency list
//    //igraph_adjlist_t adjlist_i, adjlist_j;
//    //igraph_adjlist_init(&graph_i, &adjlist_i, IGRAPH_ALL);
//    //igraph_adjlist_init(&graph_j, &adjlist_j, IGRAPH_ALL);
//
//    igraph_t& graph_i = this->graphs[i];
//    igraph_t& graph_j = this->graphs[j];
//    //compute the edge norm
//
//    //?? edge norm should be obatained from edge weight computation
//
//    //??change to the total sum of edges
//    double edge_norm1 = edge_norm;
//    double edge_norm2 = edge_norm;
//
//    //iterate through the two sets of vertices and compute the kernel values 
//    for (size_t i = 0; i < vert1.size(); i++) {
//        igraph_vector_t nei1_lab;
//        igraph_vector_init(&nei1_lab, 0);
//        igraph_vs_t vert1_nei;
//        igraph_vs_adj(&vert1_nei, vert1[i], IGRAPH_ALL);
//        igraph_cattribute_VANV(&graph_i, "label", vert1_nei, &nei1_lab);
//
//        if (igraph_vector_size(&nei1_lab) == 0) {
//            igraph_vector_destroy(&nei1_lab);
//            continue;
//        }
//        //create the nei vector
//        igraph_vector_t nei1_vec;
//        igraph_vector_init(&nei1_vec, this->n_labels);
//        for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
//            VECTOR(nei1_vec)[(int)VECTOR(nei1_lab)[m]] += edge_norm1;
//        }
//
//        std::cout << std::endl;
//        for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
//            std::cout<<VECTOR(nei1_lab)[m]<<"  ";
//        }
//
//        for (size_t j = 0; j < vert2.size(); j++) {
//            igraph_vector_t nei2_lab;
//            igraph_vector_init(&nei2_lab, 0);
//            igraph_vs_t vert2_nei;
//            igraph_vs_adj(&vert2_nei, vert2[j], IGRAPH_ALL);
//            igraph_cattribute_VANV(&graph_j, "label", vert2_nei, &nei2_lab);
//
//            if (igraph_vector_size(&nei2_lab) == 0) {
//                igraph_vector_destroy(&nei2_lab);
//                continue;
//            }
//            //create the nei vector
//            igraph_vector_t nei2_vec;
//            igraph_vector_init(&nei2_vec, this->n_labels);
//            for (size_t n = 0; n < igraph_vector_size(&nei2_lab); n++) {
//                VECTOR(nei2_vec)[(int)VECTOR(nei2_lab)[n]] += edge_norm2;
//            }
//
//            std::cout << std::endl;
//            for (size_t m = 0; m < igraph_vector_size(&nei2_lab); m++) {
//                std::cout << VECTOR(nei2_lab)[m] << "  ";
//            }
//
//            //multiple two vector and get the kernel value
//            igraph_vector_mul(&nei2_vec, &nei1_vec);
//            kernelVals.push_back(igraph_vector_sum(&nei2_vec));
//            std::cout << igraph_vector_sum(&nei2_vec) << std::endl;
//
//            igraph_vector_destroy(&nei2_lab);
//            igraph_vector_destroy(&nei2_vec);
//            igraph_vs_destroy(&vert2_nei);
//        }
//        igraph_vector_destroy(&nei1_lab);
//        igraph_vector_destroy(&nei1_vec);
//        igraph_vs_destroy(&vert1_nei);
//
//    }
//
//    //debug
//    if (!kernelVals.empty()) {
//        return *max_element(kernelVals.begin(), kernelVals.end());
//    }
//    else
//        return 0.0;
//}
//
//
//igraph_matrix_t kernel::robustKernel::kernelValTest() {
//    //count label value and construct index corresponding for adjancy matrix
//    std::set<int> label_sets;
//    int n_graphs = this->graphs.size();
//    //iterate through whole sets of graphs. and insert each graph's inverted_tree to count for the unique labels
//    for (int i = 0; i < this->inverted_indices.size(); i++) {
//        for (const auto& inv_idx : this->inverted_indices[i]) {
//            label_sets.insert(inv_idx.first);
//        }
//    }
//    //total unique labels num
//    int n_labels = label_sets.size();
//
//    //build label indices
//    std::map<int, int> label_index;
//    int idx = 0;
//    //mark each label for their positions at the label_set, and build invert index
//    for (auto i : label_sets) {
//        label_index.insert(pair<int, int>(i, idx));
//        idx++;
//    }
//
//    //Compute edge norm for each graph
//    //??
//
//    //build neighborhood label vector
//
//    //iterate the predefined iteration and compute the kernel values
//    igraph_matrix_t k_matrix;
//    igraph_matrix_init(&k_matrix, n_graphs, n_graphs);
//    igraph_matrix_null(&k_matrix);
//
//    double edge_num_sum = 0.0, edge_norm;
//    for (int i = 0; i < n_graphs; i++) {
//        edge_num_sum += this->edge_nums[i];
//    }
//    edge_norm = 1.0 / edge_num_sum;
//    for (int h = 0; h < h_max; h++) {
//        //iterate over all stored graphs
//        for (int i = 0; i < n_graphs; i++) {
//            //start from i to avoid multicount the values?? but set to start from 0 makes the computation happens twice for (i,j) and (j,i)
//            for (int j = i; j < n_graphs; j++) { //j=0 if computed from division by sum of two self-kernel values
//                auto& inv1 = this->inverted_indices[i];
//                auto& inv2 = this->inverted_indices[j];
//
//                for (auto& val : label_sets) {
//                    if (inv1.count(val) && inv2.count(val)) {
//                        auto& vers1 = inv1[val];
//                        auto& vers2 = inv2[val];
//                        double kernelval = robustKernelVal(vers1, vers2, this->graphs[i], this->graphs[j]);
//                        kernelval *= edge_norm * edge_norm;
//                        double comparKernelVal = robustKernelVal(vers1, vers1, this->graphs[i], this->graphs[i]);
//                        comparKernelVal *= edge_norm * edge_norm;
//                        if (kernelval > comparKernelVal || kernelval<0 || comparKernelVal<0) {
//                            double kernelval = robustKernelValTest(vers1, vers2, i, j, edge_norm);
//                            double comparKernelVal = robustKernelValTest(vers1, vers1, i, i, edge_norm);
//                        }
//                        if (i == j) {
//                            kernelval *= 0.5;
//                        }
//                        MATRIX(k_matrix, i, j) = MATRIX(k_matrix, i, j) + kernelval;
//                        MATRIX(k_matrix, j, i) = MATRIX(k_matrix, j, i) + kernelval;
//                    }
//                }
//            }
//        }
//    }
//
//    //get the total sum of neighboring comparison value to itself
//    //print the matrix value and check out
//    //original paper  
//
//    /*double sum_self_kernel = 0.0;
//    for (int i = 0; i < n_graphs; i++) {
//        sum_self_kernel += MATRIX(k_matrix, i, i);
//    }
//    igraph_matrix_scale(&k_matrix, 2.0/sum_self_kernel);
//    return k_matrix;*/
//
//    //use alternative equation for kernel value comparison
//    //for (int i = 0; i < n_graphs; i++) {
//    //    for (int j = 0; j < n_graphs; j++) {
//    //        std::cout << MATRIX(k_matrix, i, j) << "\t";
//    //    }
//    //    std::cout << std::endl;
//    //}
//    //double sum_self_kernel = 1.0;
//    //for (int i = 0; i < n_graphs; i++) {
//    //    sum_self_kernel *= MATRIX(k_matrix, i, i);
//    //}
//    ////loop all element and compute the score
//    //igraph_matrix_scale(&k_matrix, 1.0/sqrt(sum_self_kernel));
//    //return k_matrix;
//
//    for (int i = 0; i < n_graphs; i++) {
//        for (int j = 0; j < n_graphs; j++) {
//            std::cout << MATRIX(k_matrix, i, j) << "\t";
//        }
//        std::cout << std::endl;
//    }
//    //use the query graph as normalization
//    double sum_self_kernel = pow(MATRIX(k_matrix, 0, 0), 2);
//    //loop all element and compute the 
//    igraph_matrix_scale(&k_matrix, 1.0 / sqrt(sum_self_kernel));
//    return k_matrix;
//}






//double kernel::robustKernel::kernelValue(const vector<int>& map1, const vector<int>& map2, int& i, int& j, vector<int>& num_v, MatrixXd& node_nei)
//{
//    //calculate the kernel value increments
//    vector<double> values;
//    int start1 = 0, start2 = 0;
//    for (int m = 0; m < i; m++) {
//        start1 += num_v[m];
//    } //labels the start of ith graph
//    for (int m = 0; m < j; m++) {
//        start2 += num_v[m];
//    } //labels the start of jth graph
//
//
//    for (auto v1 : map1) {
//        for (auto v2 : map2) {
//            VectorXd label_vec1 = node_nei.row(start1 + v1);
//            VectorXd label_vec2 = node_nei.row(start2 + v2);
//
//            //compute kernel value
//            values.push_back(label_vec1.dot(label_vec2));
//        }
//
//    }
//
//    //return the largest dot product value
//    return *max_element(values.begin(), values.end());
//}
///**
// *E: the vector of edge matrix, edge = (E.ele[,0], E.ele[,1]), edge strength=E.ele[,2]
// *V_label: label values for per-graph per node, outer length V_label must equal E outer length
// *h_max: the total steps or iterations
// *V_label: label value(int) for each graph
// *num_v: num of vertices for each graph
// *num_e: num of edge for each graph
// *K_mat: kernel matrix that stores value for graphs comparison.
//
//**/
//void kernel::robustKernel::wlRobustKernel(vector<MatrixXi>& E, vector<vector<int>>& V_label, vector<int>& num_v, vector<int>& num_e, int h_max, MatrixXd& K_mat)
//{
//    //E is the vector of edge matrices, but only two graphs
//    int n = 2;                                             //num of graphs !change to E.size for multigraph processing
//    int v_all = accumulate(num_v.begin(), num_v.end(), 0); // compute the num of vertices
//    // int e_all = *max_element(num_e.begin(), num_e.end()); //find the max number of 
//
//    //insert element to the set and count the total unique values
//    set<int> n_label;
//
//    //map the label value to the index in label vector
//    std::map<int, int> label_index;
//
//    //build vector that stores the edge-normallized weight for graphs 
//    vector<VectorXd> norm_edge(n);
//
//    //build inverted index for label value
//    //vertex num system use per-graph based
//    //inverted index means label value as index and stores the vertex ind that belongs to this label
//    vector<unordered_map<int, vector<int> > > inverted_index;
//
//    for (int i = 0; i < V_label.size(); i++)
//    {
//        unordered_map<int, vector<int>> graph_inverted_index;
//        for (int j = 0; j < V_label[i].size(); j++)
//        {
//            //use label value as index
//            (graph_inverted_index[V_label[i][j]]).push_back(j);
//            n_label.insert(V_label[i][j]);
//        }
//        inverted_index.push_back(graph_inverted_index);
//    }
//    //count the num of labels
//    int n_label_vals = n_label.size();
//
//    //insert according to the labels vector's index ordered set
//    //use to mark the idx of each label value in the neighborhood label vector (see following) 
//    int idx = 0;
//    for (auto it : n_label)
//    {
//        label_index.insert(pair<int, int>(it, idx)); //sorted label values
//        idx++;
//    }
//
//    //weight the edge strength graph-wise
//    for (size_t i = 0; i < n; i++)
//    {
//        double edge_w = 0;
//        // sum of edge weights
//        edge_w = E[i].col(2).sum();
//        norm_edge[i].resize(num_v[i]);
//
//        norm_edge[i] = E[i].col(2).cast<double>();
//        norm_edge[i].array() /= edge_w;
//    }
//
//    //build neighborhood label vector for each vertex
//    MatrixXd node_nei_vec = MatrixXd::Zero(v_all, n_label_vals);
//    int raise = 0;
//    for (int i = 0; i < n; i++)
//    {
//        for (int j = 0; j < E[i].rows(); j++)
//        {
//            int edge_0_label = V_label[i][E[i](j, 0)];
//            int edge_1_label = V_label[i][E[i](j, 1)];
//
//            //add to the neighborhood list
//            //need to consider the offset for different graph
//            //label_index returns the label value's index
//            double val1 = norm_edge[i](j);
//            int val2 = E[i](j, 1);
//            int val3 = E[i](j, 0);
//            int val4 = label_index[edge_0_label];
//            node_nei_vec(E[i](j, 1) + raise, label_index[edge_0_label]) += norm_edge[i](j);
//            node_nei_vec(E[i](j, 0) + raise, label_index[edge_1_label]) += norm_edge[i](j);
//        }
//        raise += num_v[i];
//    }
//
//    //iterate the predefined iteration and update kernel value
//    K_mat.resize(n, n);
//    K_mat.setZero();
//    for (int h = 0; h < h_max; h++)
//    {
//        //iterate all graphs
//        for (int i = 0; i < n; i++)
//        {
//            int offset = 0;
//            for (int j = i; j < n; j++)
//            {
//                auto& inv_vet1 = inverted_index[i];
//                auto& inv_vet2 = inverted_index[j];
//                //TODO iterate over all label value vertex! change the vertex numbering way!
//                for (auto& val : n_label)
//                {
//                    //only increment the kernel value when same label is found in both graphs
//                    if (inv_vet1.count(val) && inv_vet2.count(val)) {
//                        auto& map1 = inv_vet1[val];
//                        auto& map2 = inv_vet2[val];
//                        double kernelInc = kernelValue(map1, map2, i, j, num_v, node_nei_vec);
//                        K_mat(i, j) += kernelInc;
//                        K_mat(j, i) += kernelInc;
//                    }
//                }
//            }
//        }
//    }
//
//    //the kernel value of each graph to itself, k_mat(i,i) is counted twice, thus need to get back
//    K_mat.diagonal() /= 2;
//
//    //normalize the kernel value matrix
//    double sum_kernel = K_mat.diagonal().sum();
//
//    //divide the kernel matrix with the total sum of self-kernel
//    K_mat /= sum_kernel;
//
//    //
//
//    //still need to divide the graph's comparison score by the total sum of neighborhood comparisons of itself.
//}
