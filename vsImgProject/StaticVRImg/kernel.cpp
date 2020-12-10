#include "kernel.h"
#include "helper.h"
#include "fileManager.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <set>
#include <bitset>
#include <omp.h>
#include <unordered_map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <igraph.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <boost/dynamic_bitset.hpp>

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

void kernel::robustKernel::push_back(igraph_t newgraph, int doc_ind) {
    igraph_t newpushgraph;
    igraph_copy(&newpushgraph, &newgraph);
    this->graphs.push_back(newpushgraph);
    this->graphPrepro(newpushgraph,doc_ind);
}

void kernel::robustKernel::graphPrepro(igraph_t& graph, int doc_ind) {
    igraph_vector_t labels;
    igraph_vector_init(&labels, 0);


    //get the label vecyor
    VANV(&graph, "label", &labels);

    size_t edgeLen = igraph_ecount(&graph);

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
        kernelval += robustKernelVal(inv_idx.second, inv_idx.second, graph, graph,doc_ind);
      
    }
    this->raw_self_kernel.push_back(kernelval);
     
    this->inverted_indices.push_back(inverted_index);
    //stores the number of vertices and edges
    this->ver_nums.push_back(igraph_vector_size(&labels));
    this->edge_nums.push_back(edgeLen);
    
    igraph_vector_destroy(&labels);
}

double kernel::robustKernel::robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, igraph_t& graph_i, igraph_t& graph_j, int doc_ind) {

    std::vector<double> kernelVals;

    //iterate through the two sets of vertices and compute the kernel values 
    for (size_t i = 0; i < vert1.size(); i++) {
        igraph_vector_t nei1_lab;
        igraph_vector_init(&nei1_lab, 0);
        igraph_vs_t vert1_nei;
        igraph_vs_adj(&vert1_nei, vert1[i], IGRAPH_ALL);
        igraph_cattribute_VANV(&graph_i, "label", vert1_nei, &nei1_lab);

        if (igraph_vector_size(&nei1_lab) == 0) { 
            igraph_vector_destroy(&nei1_lab); //if no adjacent nodes available continue to next one
            continue; 
        }

        //query the neighbor vertices and query the edge weight
        std::vector<float> nei1_vec(this->n_labels, 0);

        //depend on whether supply doc_ind to do tfidf weighting
        igraph_es_t nei_edges_i;
        igraph_es_incident(&nei_edges_i, vert1[i], IGRAPH_ALL);

        //query edge weight
        std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> nei_weights_i(new igraph_vector_t(), &igraph_vector_destroy);
        igraph_vector_init(nei_weights_i.get(), 0);
        igraph_cattribute_EANV(&graph_i, "weight", nei_edges_i, nei_weights_i.get());
        if (!this->tfidf.empty() && doc_ind != -1) {
            for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
                nei1_vec[(int)VECTOR(nei1_lab)[m]] += this->tfidf.at<float>(doc_ind, (int)VECTOR(nei1_lab)[m])* VECTOR(*nei_weights_i)[m];
            }
        }
        else
        {
            for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
                nei1_vec[(int)VECTOR(nei1_lab)[m]] += VECTOR(*nei_weights_i)[m];
            }
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
            std::vector<float> nei2_vec(this->n_labels, 0);

            igraph_es_t nei_edges_j;
            igraph_es_incident(&nei_edges_j, vert2[j], IGRAPH_ALL);
            //query edge weight
            std::unique_ptr<igraph_vector_t, void(*)(igraph_vector_t*)> nei_weights_j(new igraph_vector_t(), &igraph_vector_destroy);
            igraph_vector_init(nei_weights_j.get(), 0);
            igraph_cattribute_EANV(&graph_j, "weight", nei_edges_j, nei_weights_j.get());



            if (!this->tfidf.empty() && doc_ind != -1) {
                for (size_t n = 0; n < igraph_vector_size(&nei2_lab); n++) {
                    nei2_vec[(int)VECTOR(nei2_lab)[n]] += this->tfidf.at<float>(doc_ind, (int)VECTOR(nei2_lab)[n])* VECTOR(*nei_weights_j)[n];
                }
            }
            else
            {
                for (size_t n = 0; n < igraph_vector_size(&nei2_lab); n++) {
                    nei2_vec[(int)VECTOR(nei2_lab)[n]] +=  VECTOR(*nei_weights_j)[n];
                }
            } 

            //rebuild the vector with element-wise min
            /*std::vector<double> cwiseMinVec;
            cwiseMinVec.reserve(this->n_labels);
            std::transform(nei1_vec.begin(), nei1_vec.end(), nei2_vec.begin(), std::back_inserter(cwiseMinVec), [](float a, float b) {return std::min(a, b); });*/

            //multiple two vector and get the kernel value
            //igraph_vector_mul(&nei2_vec, &nei1_vec);
            //kernelVals.push_back(igraph_vector_sum(&nei2_vec));
            /*kernelVals.push_back(std::inner_product(cwiseMinVec.begin(), cwiseMinVec.end(), cwiseMinVec.begin(), 0.0));*/
            kernelVals.push_back(std::inner_product(nei1_vec.begin(), nei1_vec.end(), nei2_vec.begin(), 0.0));

            igraph_vector_destroy(&nei2_lab);
            igraph_vs_destroy(&vert2_nei);
        }
        igraph_vector_destroy(&nei1_lab);
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
//2th arg: indexes: sample image index
//3th arg: whether to use tfidfweight for computing
std::vector < std::vector<double>> kernel::robustKernel::robustKernelCompWithQueryArray(std::vector<igraph_t>& database_graphs, std::vector<int>* indexes, std::vector<int> *source_index, bool tfidfWeight) {
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

        //get the edge weight

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
            if(tfidfWeight&&source_index)
                kernelval += robustKernelVal(inv_idx.second, inv_idx.second, database_graphs[i], database_graphs[i],(*source_index)[i]);
            else
            {
                kernelval += robustKernelVal(inv_idx.second, inv_idx.second, database_graphs[i], database_graphs[i]);
            }
        }
        rawSelfKernel.push_back(kernelval);
        invertedIndices.push_back(inverted_index);
        //stores the number of vertices and edges
        verNums.push_back(igraph_vector_size(&labels));
        edgeNums.push_back(edgeLen / 2);

        igraph_vector_destroy(&edges);
        igraph_vector_destroy(&labels);
    }
    std::vector<std::vector<double>> k_matrix(n_query_graphs, std::vector<double>(n_database_graphs, 0.0));
    //raw score only used for debugging
    raw_scores = std::vector<std::vector<int>>(n_query_graphs, std::vector<int>(n_database_graphs, 0));
    /*igraph_matrix_t k_matrix;
    igraph_matrix_init(&k_matrix, n_query_graphs, n_database_graphs);
    igraph_matrix_null(&k_matrix);*/
    //iterate to compute each query image to database image values
    for (int h = 0; h < h_max; h++) {
        //iterate over all stored graphs
#pragma omp parallel
        {
#pragma omp for schedule(dynamic)
            for (int i = 0; i < n_query_graphs; i++) {
                //start from i to avoid multicount the values?? but set to start from 0 makes the computation happens twice for (i,j) and (j,i)
                auto& inv1 = invertedIndices[i]; //query graph

                //the index of query image in the database for tfidf score indexing if it is not nullptr
                int doc_index = indexes&&tfidfWeight ? (*indexes)[i]:-1;

                for (int j = n_query_graphs; j < n_query_graphs + n_database_graphs; j++) { //j=0 if computed from division by sum of two self-kernel values
                    auto& inv2 = invertedIndices[j]; //database graph
                    double edge_norm = 0.0;
                    if (!tfidfWeight) {
                        if (edgeNums[i] + edgeNums[j] != 0) {
                            edge_norm = 1.0 / (edgeNums[i] + edgeNums[j]);
                        }
                        else {
                            edge_norm = 0.0;
                        }
                    }                   
                    for (auto& val : labelSets) {
                        if (inv1.count(val) && inv2.count(val)) {
                            auto& vers1 = inv1[val];
                            auto& vers2 = inv2[val];
                            double kernelval = robustKernelVal(vers1, vers2, this->graphs[i], database_graphs[j- n_query_graphs], doc_index);
                            
                            //raw_scores is only for debug
                            raw_scores[i][j - n_query_graphs] = raw_scores[i][j - n_query_graphs] + kernelval;
                            if (!tfidfWeight) {
                                kernelval *= edge_norm * edge_norm;
                            }
                            /*auto ptr = igraph_matrix_e_ptr(&k_matrix, 1, 0);
                            *ptr = *ptr+kernelval;
                            }
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
    helper::computeScore1(k_matrix, edgeNums, rawSelfKernel,tfidfWeight);
    /*helper::computeScore3(k_matrix, rawSelfKernel);*/
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

bool kernel::covisMap::existEntry(int image_id) {
    return image_id < invert_index_.size()-1 && !invert_index_[image_id].empty();
}

boost::dynamic_bitset<uint8_t> kernel::covisMap::getEntry(int image_id) {
    return invert_index_[image_id];
}

void kernel::covisMap::addEntry(int image_id, colmap::FeatureVisualIDs& id) {
    //add id to invert_index
    if (id.dictsize != this->numCenters_) {
        std::cerr << "covisMap.addEntry: error: the entry vocab size unmatch the object.\n";
        return;
    }
    if (image_id > invert_index_.size()-1) {//imageId start with 1
        invert_index_.resize(image_id+1);
    }
    invert_index_[image_id].reset();
    invert_index_[image_id].resize(numCenters_);
    
    //loop through the ids and add to the invert index
    for (Eigen::Index i = 0; i < id.ids.rows(); i++) {
        invert_index_[image_id].set(id.ids(i, 1));
    }
   
}
void kernel::covisMap::Query(colmap::FeatureVisualIDs& qryId, std::vector<int>& candids) {
    //compare with the whole map and find the candidate ids
    candids.clear();
    if (qryId.dictsize != numCenters_) {
        std::cerr << "covisMap.Query: error: query id vocab size doesn't match the object vocab\n";
        return;
    }
    boost::dynamic_bitset<uint8_t> clique(numCenters_);
    clique.reset();
    for (Eigen::Index i = 0; i < qryId.ids.rows(); i++) {
        clique.set(qryId.ids(i, 1));
    }

    //search on the invert_index_
    for(int i = 0; i < invert_index_.size(); i++) {
        //bitset AND keep common words only
        auto rel = invert_index_[i] & clique;
        //check the common words threshold
        if ((float)rel.count() / clique.count() > params::PCommonwords) {
            candids.push_back(i);
        }
    }
}

/*** CAUTION: below covisMap member functions are deprecated ***/
/*******first the graph class build fully-connected graph and push it to the vector covisMap then build inverted index and cliques from the map******/
void kernel::covisMap::process(igraph_t& mygraph) {
    //process the graph, build covismap
    igraph_vector_t labs;
    igraph_vector_init(&labs, 0);
    VANV(&mygraph, "label", &labs);
    //special for the first graph
    int ncols = igraph_sparsemat_ncol(&this->map);
    std::cout << "number of cols: " << ncols << std::endl;
    std::vector<bool> unique_labs(this->numCenters_,true);
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
    std::vector<bool> unique_labs(this->numCenters_,true);
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
    //check for minimum condidate condition: only consider the common percent of unique visual ids
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

void kernel::covisMap::processSampleLoc(){

}

/****************************/
/****************************/
/***** CLASS RECURROBUSTKEL ******/
/****************************/
/****************************/

kernel::recurRobustKel::recurRobustKel(int h_max, size_t n_labels):h_max(h_max), n_labels(n_labels) {

}
kernel::recurRobustKel::~recurRobustKel() {
    for (auto& g : this->graphs) {
        igraph_destroy(&g);
    }
}
void kernel::recurRobustKel::push_back(igraph_t newgraph) {
    igraph_t newpushgraph;
    igraph_copy(&newpushgraph, &newgraph);
    this->graphs.push_back(newpushgraph);
    this->graphPrepro(newpushgraph);
}

void kernel::recurRobustKel::graphPrepro(igraph_t& graph) {
    igraph_vector_t labels;
    igraph_vector_init(&labels, 0);

    //get the label vecyor
    VANV(&graph, "label", &labels);

    //build inverted indices
    unordered_map<int, vector<size_t>> inverted_index;
    for (size_t i = 0; i < igraph_vector_size(&labels); i++) {
        inverted_index[VECTOR(labels)[i]].push_back(i);
    }

    this->inverted_indices.push_back(inverted_index);

    //stores the number of vertices and edges
    this->ver_nums.push_back(igraph_vector_size(&labels));
    this->edge_nums.push_back(igraph_ecount(&graph));

    igraph_vector_destroy(&labels);
}

void kernel::recurRobustKel::robustKernelCom(int i, int j, kernel::scoreType& kernel_vals, vetIndType& inv1, vetIndType& inv2) {
    //define the pool and 
    //iterate over all stored graphs
    std::vector<std::bitset<30000>> neighbor_vectors_i, neighbor_vectors_j;
    //build bits neighborhood vector for every nodes of the comparison graphs
    //set bit operation
    neighbor_vectors_i.resize(this->ver_nums[i]);
    neighbor_vectors_j.resize(this->ver_nums[j]);
            
    //extract inverted index
    inv1 = vetIndType(this->inverted_indices[i]);
    inv2 = vetIndType(this->inverted_indices[j]);
        
    for (int h = 0; h < h_max; h++) {
        //records the pairs of same labels and do iterateions
        if (h == 0) { 
            //use first graph to count for the common nodes
            for (auto val : inv1) {
                //init neighborhood vector
                auto vers1 = val.second;
                //mark the label on the label vector which for h=0 means label itself
                for (auto k : vers1) {
                    neighbor_vectors_i[k].set(val.first, true);
                }
                        
                if (inv2.count(val.first)) {                      
                    //add first round kernel values, we use maximum kernel values for a label
                    kernel_vals[val.first].push_back(1.0);
                }
                else
                {
                    inv1.erase(val.first); //erase those keys without matching items
                }
            }
            for (auto val : inv2) {
                auto vers2 = val.second;
                for (auto k : vers2) {
                    neighbor_vectors_j[k].set(val.first, true);
                }
            }
        }

        //other than 0 round do iterative kernel computing
        if (h != 0) {
            //update the neighborhood vector with its neighbors
            igraph_vector_t nei1_lab;
            igraph_vector_init(&nei1_lab, 0);
            auto copy_neighbor_vectors_i = neighbor_vectors_i;
            auto copy_neighbor_vectors_j = neighbor_vectors_j;

            //update ith graph's neighbor_vector
            for (auto val : this->inverted_indices[i]) {
                for (auto vert : val.second) {
                    igraph_neighbors(&this->graphs[i], &nei1_lab, vert, IGRAPH_ALL);

                    //update each vertex neighbor vector
                    for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                        //XOR set neighbor vector
                        copy_neighbor_vectors_i[vert]|= neighbor_vectors_i[VECTOR(nei1_lab)[k]];
                    }
                }
            }
            //update jth graph's neighbor_vector
            for (auto val : this->inverted_indices[j]) {
                for (auto vert : val.second) {
                    igraph_neighbors(&this->graphs[j], &nei1_lab, vert, IGRAPH_ALL);

                    //update each vertex neighbor vector
                    for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                        //binary OR to set neighbor vector
                        copy_neighbor_vectors_j[vert] |= neighbor_vectors_j[VECTOR(nei1_lab)[k]];
                    }
                }
            }
            //update neighbor vectors
            neighbor_vectors_i = copy_neighbor_vectors_i;
            neighbor_vectors_j= copy_neighbor_vectors_j;
            copy_neighbor_vectors_i.clear();
            copy_neighbor_vectors_j.clear();
            //compute the kernel value for all common nodes and filter those unimportant matches
            for (auto val : inv1) {
                std::set<int> seti, setj;
                auto ver2 = inv2[val.first];

                //keep track the kernel value from last round
                int maxkel=kernel_vals[val.first].back();
                for (auto vert1 : val.second) {
                    for (auto vert2 : ver2) {
                        //if the neighbor vector dot product increases then preserve it
                        int kel = (neighbor_vectors_i[vert1] & neighbor_vectors_j[vert2]).count();

                        //strict condition: kel> kernel_vals[val.first].back(), delete those weight unchanged nodes 
                        if ( kel >= h + 1)
                        {
                            maxkel = kel > maxkel ? kel : maxkel;
                            seti.insert(vert1);
                            setj.insert(vert2);
                        }
                    }
                }
                //update neighbor vector for both graphs
                inv1[val.first] = std::vector<size_t>(seti.begin(),seti.end());
                inv2[val.first] = std::vector<size_t>(setj.begin(), setj.end());
                kernel_vals[val.first].push_back(maxkel);
            }
            igraph_vector_destroy(&nei1_lab);
        }
                
    }
    neighbor_vectors_i.clear();
    neighbor_vectors_j.clear();
}

void kernel::recurRobustKel::robustKernelCom(int i, igraph_t& source_graph, scoreType& kernel_vals, vetIndType& inv1, vetIndType& inv2, bool useTFIDF) {
    /***preprocessing the source graph***/
    igraph_vector_t labels;
    igraph_vector_init(&labels, 0);

    //get the label vecyor
    VANV(&source_graph, "label", &labels);

    //build inverted indices
    unordered_map<int, vector<size_t>> source_inverted_index;
    for (size_t i = 0; i < igraph_vector_size(&labels); i++) {
        source_inverted_index[VECTOR(labels)[i]].push_back(i);
    }

    //stores the number of vertices and edges
    int source_num_vets = igraph_vcount(&source_graph);
    int source_num_edges = igraph_ecount(&source_graph);

    igraph_vector_destroy(&labels);
       
    /**define the pool and 
    iterate over all stored graphs**/
    //change to dynamic bits
    /*boost::dynamic_bitset<> neighbor_vector_i(this->n_labels);*/
    std::vector<std::bitset<100000>> neighbor_vectors_i, neighbor_vectors_j;
    //build bits neighborhood vector for every nodes of the comparison graphs
    //set bit operation
    neighbor_vectors_i.resize(this->ver_nums[i]);
    neighbor_vectors_j.resize(source_num_vets);

    //extract inverted index
    inv1.clear();
    /*inv2 = vetIndType(source_inverted_index);*/

    inv2 = source_inverted_index;

    //inv1 and source_inverted_index are used as pools, which will be modified during the iteration
    for (int h = 0; h < h_max; h++) {
        //records the pairs of identical labels and build pools for iteration
        if (h == 0) {
            //use first graph to count for the common nodes, the ith query graph
            for (auto &val : this->inverted_indices[i]) {
                //init neighborhood vector
                auto vers1 = val.second;
                
                //mark the label in dictionary vector for each No.k nodes which for h=0 means label itself
                for (auto k : vers1) {
                    neighbor_vectors_i[k].set(val.first, true);
                }
                //if it is the common nodes
                if (source_inverted_index.count(val.first)) {
                    //inv1 is served as container for the ith query graph pool and kernel_vals is the score recorder for all common nodes kernel value
                    inv1[val.first] = vers1; //inv1.first is the common node ids but .second is graph i inverted indices
                    kernel_vals[val.first].push_back(1.0);
                }
            }
            //iterate over the graph j inverted index
            for (auto val : inv2) {
                for (auto k : val.second) {
                    neighbor_vectors_j[k].set(val.first, true);
                }
            }
        }

        //other than 0 round do iterative kernel computing
        if (h != 0) {
            //update the neighborhood vector with its neighbors
            igraph_vector_t nei1_lab;
            igraph_vector_init(&nei1_lab, 0);
            auto copy_neighbor_vectors_i = neighbor_vectors_i;
            auto copy_neighbor_vectors_j = neighbor_vectors_j;

            //update ith graph's neighbor_vector
            for (auto val : this->inverted_indices[i]) {
                for (auto vert : val.second) {
                    igraph_neighbors(&this->graphs[i], &nei1_lab, vert, IGRAPH_ALL);

                    //update each vertex neighbor vector
                    for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                        //OR set neighbor vector
                        copy_neighbor_vectors_i[vert] |= neighbor_vectors_i[VECTOR(nei1_lab)[k]];
                    }
                }
            }
            //update jth graph's neighbor_vector
            for (auto val : source_inverted_index) {
                for (auto vert : val.second) {
                    igraph_neighbors(&source_graph, &nei1_lab, vert, IGRAPH_ALL);

                    //update each vertex neighbor vector
                    for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                        //binary OR to set neighbor vector
                        copy_neighbor_vectors_j[vert] |= neighbor_vectors_j[VECTOR(nei1_lab)[k]];
                    }
                }
            }
            //update neighbor vectors
            neighbor_vectors_i = copy_neighbor_vectors_i;
            neighbor_vectors_j = copy_neighbor_vectors_j;
            copy_neighbor_vectors_i.clear();
            copy_neighbor_vectors_j.clear();
            //compute the kernel value for all common nodes and filter those unimportant matches
            for (auto val : inv1) {
                std::set<int> seti, setj;
                auto ver2 = inv2[val.first];

                //keep track the kernel value from last round
                int maxkel = kernel_vals[val.first].back();
                for (auto vert1 : val.second) {
                    for (auto vert2 : ver2) {
                        //if the neighbor vector dot product increases then preserve it
                        int kel = (neighbor_vectors_i[vert1] & neighbor_vectors_j[vert2]).count();

                        //strict condition: kel> kernel_vals[val.first].back(), delete those weight unchanged nodes 
                        if (kel >= h + 1)
                        {
                            maxkel = kel > maxkel ? kel : maxkel;
                            seti.insert(vert1);
                            setj.insert(vert2);
                        }
                    }
                }
                //update neighbor vector for both graphs
                inv1[val.first] = std::vector<size_t>(seti.begin(), seti.end());
                inv2[val.first] = std::vector<size_t>(setj.begin(), setj.end());
                kernel_vals[val.first].push_back(maxkel);
            }
            igraph_vector_destroy(&nei1_lab);
        }

    }
    neighbor_vectors_i.clear();
    neighbor_vectors_j.clear();
}

void kernel::recurRobustKel::robustKernelCom(igraph_t& query_graph, igraph_t& source_graph, scoreType& kernel_vals, vetIndType& inv1, vetIndType& inv2, bool useTFIDF) {
    /***preprocessing the source graph***/
    igraph_vector_t labels, query_labels;
    igraph_vector_init(&labels, 0);
    igraph_vector_init(&query_labels, 0);

    //get the label vecyor
    VANV(&source_graph, "label", &labels);
    VANV(&query_graph, "label", &query_labels);

    //build inverted indices
    unordered_map<int, vector<size_t>> source_inverted_index, query_inverted_index;
    for (size_t i = 0; i < igraph_vector_size(&labels); i++) {
        source_inverted_index[VECTOR(labels)[i]].push_back(i);
    }
    for (size_t i = 0; i < igraph_vector_size(&query_labels); i++) {
        query_inverted_index[VECTOR(query_labels)[i]].push_back(i);
    }
    

    //set up a intermediate graph for recording the kernel process
    igraph_t query_graph_record, source_graph_record;
    igraph_empty(&query_graph_record,0,IGRAPH_UNDIRECTED);
    igraph_empty(&source_graph_record, 0, IGRAPH_UNDIRECTED);

    //records all pairs of nodes
    std::map<int, int> nodeMapQuery, nodeMapSource;

    //stores the number of vertices and edges
    int source_num_vets = igraph_vcount(&source_graph);
    int source_num_edges = igraph_ecount(&source_graph);

    int query_num_vets = igraph_vcount(&query_graph);
    int query_num_edges = igraph_ecount(&query_graph);

    igraph_vector_destroy(&labels);
    igraph_vector_destroy(&query_labels);

    /**define the pool and
    iterate over all stored graphs**/
    std::vector<std::bitset<30000>> neighbor_vectors_i, neighbor_vectors_j;
    //build bits neighborhood vector for every nodes of the comparison graphs
    //set bit operation
    neighbor_vectors_i.resize(query_num_vets);
    neighbor_vectors_j.resize(source_num_vets);

    //extract inverted index
    inv1.clear();
    inv2 = vetIndType(source_inverted_index);

    for (int h = 0; h < h_max; h++) {
        //records the pairs of same labels and do iterateions
        if (h == 0) {
            //use first graph to count for the common nodes
            for (auto& val : query_inverted_index) {
                //init neighborhood vector
                std::vector<size_t> vers1 = val.second;
                //if it is the common nodes
                if (source_inverted_index.count(val.first)) {
                    //add first round kernel values, we use maximum kernel values for a label
                    inv1[val.first] = vers1;
                    kernel_vals[val.first].push_back(1.0);

                    //add to both graphs
                    int query_vcount = igraph_vcount(&query_graph_record);
                    igraph_add_vertices(&query_graph_record, val.second.size(),0);
                    for (int k = 0; k < val.second.size(); k++) {
                        SETVAN(&query_graph_record, "posx", query_vcount +k, VAN(&query_graph, "posx", val.second[k]));
                        SETVAN(&query_graph_record, "posy", query_vcount + k, VAN(&query_graph, "posy", val.second[k]));
                        SETVAN(&query_graph_record, "label", query_vcount + k, VAN(&query_graph, "label", val.second[k]));
                        SETVAN(&query_graph_record, "h", query_vcount + k, h);
                        nodeMapQuery.insert({ val.second[k] , query_vcount + k });
                    }
                    auto vers2 = source_inverted_index.at(val.first);
                    int source_vcount = igraph_vcount(&source_graph_record);
                    igraph_add_vertices(&source_graph_record, vers2.size(), 0);
                    for (int k = 0; k < vers2.size(); k++) {
                        SETVAN(&source_graph_record, "posx", source_vcount + k, VAN(&source_graph, "posx", vers2[k]));
                        SETVAN(&source_graph_record, "posy", source_vcount + k, VAN(&source_graph, "posy", vers2[k]));
                        SETVAN(&source_graph_record, "label", source_vcount + k, VAN(&source_graph, "label", vers2[k]));
                        SETVAN(&source_graph_record, "h", source_vcount + k, h);
                        nodeMapSource.insert({ vers2[k] , source_vcount + k });
                    }
                }
                //mark the label on the label vector which for h=0 means label itself 
                for (int k = 0; k < vers1.size(); k++) {
                    neighbor_vectors_i[vers1[k]].set(val.first, true);
                }


            }
            for (auto val : inv2) {
                auto vers2 = val.second;
                for (auto k : vers2) {
                    neighbor_vectors_j[k].set(val.first, true);
                }
            }
        }

        //other than 0 round do iterative kernel computing
        if (h != 0) {
            //update the neighborhood vector with its neighbors
            igraph_vector_t nei1_lab;
            igraph_vector_init(&nei1_lab, 0);
            auto copy_neighbor_vectors_i = neighbor_vectors_i;
            auto copy_neighbor_vectors_j = neighbor_vectors_j;

            //update ith graph's neighbor_vector
            for (auto val : query_inverted_index) {
                for (auto vert : val.second) {
                    igraph_neighbors(&query_graph, &nei1_lab, vert, IGRAPH_ALL);

                    //update each vertex neighbor vector
                    for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                        //XOR set neighbor vector
                        copy_neighbor_vectors_i[vert] |= neighbor_vectors_i[VECTOR(nei1_lab)[k]];
                    }
                }
            }
            //update jth graph's neighbor_vector
            for (auto val : source_inverted_index) {
                for (auto vert : val.second) {
                    igraph_neighbors(&source_graph, &nei1_lab, vert, IGRAPH_ALL);

                    //update each vertex neighbor vector
                    for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                        //binary OR to set neighbor vector
                        copy_neighbor_vectors_j[vert] |= neighbor_vectors_j[VECTOR(nei1_lab)[k]];
                    }
                }
            }
            //update neighbor vectors
            neighbor_vectors_i = copy_neighbor_vectors_i;
            neighbor_vectors_j = copy_neighbor_vectors_j;
            copy_neighbor_vectors_i.clear();
            copy_neighbor_vectors_j.clear();
            //compute the kernel value for all common nodes and filter those unimportant matches
            for (auto val : inv1) {
                std::set<int> seti, setj;
                auto ver2 = inv2[val.first];

                //keep track the kernel value from last round
                int maxkel = kernel_vals[val.first].back();
                for (auto vert1 : val.second) {
                    for (auto vert2 : ver2) {
                        //if the neighbor vector dot product increases then preserve it
                        int kel = (neighbor_vectors_i[vert1] & neighbor_vectors_j[vert2]).count();

                        //strict condition: kel> kernel_vals[val.first].back(), delete those weight unchanged nodes 
                        if (kel >= h + 1)
                        {
                            maxkel = kel > maxkel ? kel : maxkel;
                            seti.insert(vert1);
                            setj.insert(vert2);
                        }
                        //exclude the vertex that is not qualified and set them to different color
                        else {
                            SETVAN(&query_graph_record, "h", nodeMapQuery[vert1], 10.0);
                            SETVAN(&source_graph_record, "h", nodeMapSource[vert2], 10.0);
                        }
                    }
                }
                //update neighbor vector for both graphs
                inv1[val.first] = std::vector<size_t>(seti.begin(), seti.end());
                inv2[val.first] = std::vector<size_t>(setj.begin(), setj.end());
                kernel_vals[val.first].push_back(maxkel);
            }
            igraph_vector_destroy(&nei1_lab);

            //update the record graph with neighborhoolds
            for (const auto& keyVals : nodeMapQuery) {
                igraph_vector_t nei1_lab;
                igraph_vector_init(&nei1_lab, 0);
                igraph_neighbors(&query_graph, &nei1_lab, keyVals.first, IGRAPH_ALL);

                //add neighbors to both graphs

                for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                    if (!nodeMapQuery.count(VECTOR(nei1_lab)[k])) {
                        //add edge and add vertex
                        int query_vcount = igraph_vcount(&query_graph_record);
                        igraph_add_vertices(&query_graph_record, 1, 0);
                        igraph_add_edge(&query_graph_record, keyVals.second, query_vcount);
                        SETVAN(&query_graph_record, "posx", query_vcount, VAN(&query_graph, "posx", VECTOR(nei1_lab)[k]));
                        SETVAN(&query_graph_record, "posy", query_vcount, VAN(&query_graph, "posy", VECTOR(nei1_lab)[k]));
                        SETVAN(&query_graph_record, "label", query_vcount, VAN(&query_graph, "label", VECTOR(nei1_lab)[k]));
                        SETVAN(&query_graph_record, "h", query_vcount, h);
                        nodeMapQuery.insert({ VECTOR(nei1_lab)[k] , query_vcount });
                    }
                    else {
                        igraph_add_edge(&query_graph_record, keyVals.second, nodeMapQuery.at(VECTOR(nei1_lab)[k]));
                    }
                }
            }
            for (const auto& keyVals : nodeMapSource) {
                //second graphs add neighborhood
                igraph_vector_t nei2_lab;
                igraph_vector_init(&nei2_lab, 0);
                igraph_neighbors(&source_graph, &nei2_lab, keyVals.first, IGRAPH_ALL);
                
                for (int k = 0; k < igraph_vector_size(&nei2_lab); k++) { 
                    if (!nodeMapSource.count(VECTOR(nei2_lab)[k])) {
                        int source_vcount = igraph_vcount(&source_graph_record);
                        igraph_add_vertices(&source_graph_record, 1, 0);
                        igraph_add_edge(&source_graph_record, keyVals.second, source_vcount);
                        SETVAN(&source_graph_record, "posx", source_vcount, VAN(&source_graph, "posx", VECTOR(nei2_lab)[k]));
                        SETVAN(&source_graph_record, "posy", source_vcount, VAN(&source_graph, "posy", VECTOR(nei2_lab)[k]));
                        SETVAN(&source_graph_record, "label", source_vcount, VAN(&source_graph, "label", VECTOR(nei2_lab)[k]));
                        SETVAN(&source_graph_record, "h", source_vcount, h);
                        nodeMapSource.insert({ VECTOR(nei2_lab)[k] , source_vcount });
                    }
                    else {
                        igraph_add_edge(&source_graph_record, keyVals.second, nodeMapSource[VECTOR(nei2_lab)[k]]);
                    }
                }
                
            }
        }

    }
    igraph_attribute_combination_t comb;
    igraph_attribute_combination(&comb,
        "weight", IGRAPH_ATTRIBUTE_COMBINE_SUM,
        "", IGRAPH_ATTRIBUTE_COMBINE_FIRST,
        IGRAPH_NO_MORE_ATTRIBUTES);

    igraph_simplify(&source_graph_record, 0, 1, &comb);
    igraph_simplify(&query_graph_record, 0, 1, &comb);
    igraph_attribute_combination_destroy(&comb);
    fileManager::write_graph(source_graph_record, "source_graph","graphml");
    fileManager::write_graph(query_graph_record, "query_graph", "graphml");
    neighbor_vectors_i.clear();
    neighbor_vectors_j.clear();
    igraph_destroy(&source_graph_record);
    igraph_destroy(&query_graph_record);
}

void kernel::recurRobustKel::robustKernelCom(int i, igraph_t& source_graph, scoreType& kernel_vals) {
    /***preprocessing the source graph***/
    igraph_vector_t labels_i, labels_j;

    igraph_vector_init(&labels_i, 0);
    igraph_vector_init(&labels_j, 0);

    //get the label vecyor
    VANV(&this->graphs[i], "label", &labels_i);
    VANV(&source_graph, "label", &labels_j);
    

    //build inverted indices, which will be used to filter the nodes with common labels
   /* unordered_map<int, vector<size_t>> source_inverted_index;*/
    vetIndType inv1, inv2;
    for (size_t item = 0; item < igraph_vector_size(&labels_j); item++) {
        inv2[VECTOR(labels_j)[item]].push_back(item);
    }

    //stores the number of vertices and edges
    int source_num_vets = igraph_vcount(&source_graph);
    int source_num_edges = igraph_ecount(&source_graph);

    /*igraph_vector_destroy(&labels_j);*/

    /**define the pool and
    iterate over all stored graphs**/
    std::vector<std::bitset<100000>> neighbor_vectors_i, neighbor_vectors_j;
    //build bits neighborhood vector for every nodes of the comparison graphs
    //set bit operation
    neighbor_vectors_i.resize(this->ver_nums[i]);
    neighbor_vectors_j.resize(source_num_vets);

    //extract inverted index
    inv1.clear();

    //inv1 and source_inverted_index are used as pools, which will be modified during the iteration
    for (int h = 0; h < h_max; h++) {
        //records the pairs of identical labels and build pools for iteration
        if (h == 0) {
            //use first graph to count for the common nodes, the ith query graph
            std::bitset<100000> graph_i_labels, graph_j_labels;
            for (int item = 0; item < igraph_vector_size(&labels_i); item++) {
                neighbor_vectors_i[item].set(VECTOR(labels_i)[item], true);
                graph_i_labels.set(VECTOR(labels_i)[item], true);
            }

            for (int j = 0; j < igraph_vector_size(&labels_j); j++) {
                neighbor_vectors_j[j].set(VECTOR(labels_j)[j], true);
                graph_j_labels.set(VECTOR(labels_j)[j], true);
            }
            //delete labels_i and labels_j since we don't need it anymore
            igraph_vector_destroy(&labels_i);
            igraph_vector_destroy(&labels_j);

            //compute the common labels;
            graph_i_labels &= graph_j_labels;

            //create pools for common nodes
            for (int k = 0; k < this->n_labels; k++) {
                if (graph_i_labels.test(k)) {
                    inv1[k] = this->inverted_indices[i][k]; //label k shown case
                    kernel_vals[k].push_back(1.0); //record kernel vals;
                }
            }


            //for (auto& val : this->inverted_indices[i]) {
            //    //init neighborhood vector
            //    auto vers1 = val.second;
            //    //mark the label in dictionary vector for each No.k nodes which for h=0 means label itself
            //    for (auto k : vers1) {
            //        neighbor_vectors_i[k].set(val.first, true);
            //    }
            //    //if it is the common nodes
            //    if (source_inverted_index.count(val.first)) {
            //        //inv1 is served as container for the ith query graph pool and kernel_vals is the score recorder for all common nodes kernel value
            //        inv1[val.first] = vers1; //inv1.first is the common node ids but .second is graph i inverted indices
            //        kernel_vals[val.first].push_back(1.0);
            //    }
            //}
            ////iterate over the graph j inverted index
            //for (auto val : inv2) {
            //    for (auto k : val.second) {
            //        neighbor_vectors_j[k].set(val.first, true);
            //    }
            //}
        }

        //other than 0 round do iterative kernel computing
        if (h != 0) {
            //update the neighborhood vector with its neighbors
            igraph_vector_t nei1_lab;
            igraph_vector_init(&nei1_lab, 0);
            auto copy_neighbor_vectors_i = neighbor_vectors_i;
            auto copy_neighbor_vectors_j = neighbor_vectors_j;

            //iterate each nodes and update with neighbors
            for (int item = 0; item < this->ver_nums[i]; item++) {
                igraph_neighbors(&this->graphs[i], &nei1_lab, item, IGRAPH_ALL);
                //update neighbor vector
                for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                    copy_neighbor_vectors_i[item] |= neighbor_vectors_i[VECTOR(nei1_lab)[k]];
                }
            }
            for (int j = 0; j < source_num_vets; j++) {
                igraph_neighbors(&source_graph, &nei1_lab, j, IGRAPH_ALL);
                for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
                    copy_neighbor_vectors_j[j] |= neighbor_vectors_j[VECTOR(nei1_lab)[k]];
                }
            }

            //compute kernel value and update the common label vector



            //update ith graph's neighbor_vector
            //for (auto val : this->inverted_indices[i]) {
            //    for (auto vert : val.second) {
            //        igraph_neighbors(&this->graphs[i], &nei1_lab, vert, IGRAPH_ALL);

            //        //update each vertex neighbor vector
            //        for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
            //            //OR set neighbor vector
            //            copy_neighbor_vectors_i[vert] |= neighbor_vectors_i[VECTOR(nei1_lab)[k]];
            //        }
            //    }
            //}
            ////update jth graph's neighbor_vector
            //for (auto val : source_inverted_index) {
            //    for (auto vert : val.second) {
            //        igraph_neighbors(&source_graph, &nei1_lab, vert, IGRAPH_ALL);

            //        //update each vertex neighbor vector
            //        for (int k = 0; k < igraph_vector_size(&nei1_lab); k++) {
            //            //binary OR to set neighbor vector
            //            copy_neighbor_vectors_j[vert] |= neighbor_vectors_j[VECTOR(nei1_lab)[k]];
            //        }
            //    }
            //}
            //update neighbor vectors
            neighbor_vectors_i = copy_neighbor_vectors_i;
            neighbor_vectors_j = copy_neighbor_vectors_j;
            copy_neighbor_vectors_i.clear();
            copy_neighbor_vectors_j.clear();
            //compute the kernel value for all common nodes and filter those unimportant matches
            for (auto val : inv1) {
                if (val.second.empty()) {
                    continue;
                }
                std::set<int> seti, setj;
                auto ver2 = inv2[val.first];
                //keep track the kernel value from last round
                int maxkel = kernel_vals[val.first].back(); //val.first represents this label value from the last round
                for (auto vert1 : val.second) { //each vertex with the same label
                    for (auto vert2 : ver2) {
                        //if the neighbor vector dot product(kernel val) is increasing then reserve it
                        int kel = (neighbor_vectors_i[vert1] & neighbor_vectors_j[vert2]).count();

                        //strict condition: kel> kernel_vals[val.first].back(), delete those weight unchanged nodes 
                        if (kel >= h + 1)
                        {
                            maxkel = kel > maxkel ? kel : maxkel;
                            seti.insert(vert1);
                            setj.insert(vert2);
                        }
                    }
                }
                //update neighbor vector for both graphs
                inv1[val.first] = std::vector<size_t>(seti.begin(), seti.end());
                inv2[val.first] = std::vector<size_t>(setj.begin(), setj.end());
                kernel_vals[val.first].push_back(maxkel);
            }
            igraph_vector_destroy(&nei1_lab);
        }

    }
    neighbor_vectors_i.clear();
    neighbor_vectors_j.clear();
}
std::vector<std::vector<float>> kernel::recurRobustKel::robustKernelCompWithQueryArray(std::vector<igraph_t>& database_graphs, std::vector<int>* source_indexes) {
    //count label value and construct index corresponding for adjancy matrix
    /*std::set<int> label_sets;*/
    int n_query_graphs = this->graphs.size();
    int n_database_graphs = database_graphs.size();

    //preprocessing database graphs
    //!!!DIDN"T PROCESS LABELSET TO COMPUTE THE TOTAL UNIQUE LABELS
    /*for (int i = 0; i < database_graphs.size(); i++) {
        this->push_back(database_graphs[i]);
    }*/

    std::vector<std::vector<float>> k_matrix(n_query_graphs, std::vector<float>(n_database_graphs, 0.0));
    //raw score only used for debugging
    //omp_set_num_threads(6);
    ////iterate to compute each query image to database image values
    // //iterate over all stored graphs
    //#pragma omp parallel
    //{
    //    #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_query_graphs; i++) {
            //start from i to avoid multicount the values?? but set to start from 0 makes the computation happens twice for (i,j) and (j,i)
            /*for (int j = n_query_graphs; j < n_query_graphs + n_database_graphs; j++) {*/ //j=0 if computed from division by sum of two self-kernel values
            for (int j = 0; j < n_database_graphs; j++) {
                scoreType scores;
                vetIndType vet1, vet2;
                /*robustKernelCom(i, database_graphs[j], scores, vet1, vet2);*/
                robustKernelCom(i, database_graphs[j], scores);
                //simple compute the total score
                for (auto& val : scores) {
                    //start of score matrix 0 is n_query_graphs in inverted_index vector
                    if (!tfidf.empty()&&source_indexes!=nullptr)
                    {
                        k_matrix[i][j] += val.second.back()*(this->tfidf.at<float>((*source_indexes)[j],val.first));
                    }
                    else
                    {
                        k_matrix[i][j] += val.second.back();
                    }
                }
            }
        }
    //}

    //clear the database graph information and only keeps the query graphs information
    /*clearDatabaseGraphs(n_query_graphs);*/
    return k_matrix;
}

void kernel::recurRobustKel::clearDatabaseGraphs(int n_query_graphs) {
    int totalSize = this->graphs.size();
    if(totalSize<n_query_graphs){
        std::cout << "ERROR: cleargraph function cannot handle query graphs less bigger than total size.";
        return;
    }
    this->edge_nums.resize(n_query_graphs);
    this->ver_nums.resize(n_query_graphs);
    this->inverted_indices.resize(n_query_graphs);
    for (int i = n_query_graphs; i < totalSize; i++) {
        igraph_destroy(&this->graphs[i]);
    }
    this->graphs.resize(n_query_graphs);
}


double kernel::recurRobustKel::robustKernelVal(std::vector<size_t>& vert1, std::vector<size_t>& vert2, igraph_t& graph_i, igraph_t& graph_j, int doc_ind) {

    std::vector<double> kernelVals;

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
        std::vector<double> nei1_vec(this->n_labels, 0);
        //igraph_vector_t nei1_vec;
        //depend on whether supply doc_ind to do tfidf weighting

        for (size_t m = 0; m < igraph_vector_size(&nei1_lab); m++) {
            nei1_vec[(int)VECTOR(nei1_lab)[m]] += 1;

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

                    //rebuild the vector with element-wise min
                    std::vector<double> cwiseMinVec;
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
    }
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
