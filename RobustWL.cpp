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
/**
 *E: the vector of edge matrix, edge = (E[,0], E[,1]), edge strength=E[,2]
 *V_label: label values for per-graph per node, outer length V_label must equal E outer length
 *h_max: the total steps or iterations
 *

**/

void WLRobustKernel(vector<MatrixXi> &E, vector<vector<int> > &V_label, vector<int> &num_v, vector<int> &num_e, int h_max, MatrixXd &K_mat)
{
    //E is the vector of edge matrices, but only two graphs
    K_mat.setZero();
    int n = 2;                                             //num of graphs !change to E.size for multigraph processing
    int v_all = accumulate(num_v.begin(), num_v.end(), 0); // compute the num of vertices
    int e_all = accumulate(num_e.begin(), num_e.end(), 0);

    //insert element to the set and count the total unique values
    set<int> n_label;

    //map the label value to the index in label vector
    std::map<int, int> label_index;

    MatrixXi norm_edge(e_all, n); // normalized edge weights

    //build inverted index for label value
    //vertex num system use per-graph based
    vector<unordered_map<int, vector<int> > > inverted_index;

    for (int i = 0; i < V_label.size(); i++)
    {
        unordered_map<int, vector<int> > graph_inverted_index;
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


    //insert according to the labels vector's index
    int idx = 0;
    for(auto it:n_label){
        label_index.insert(pair<int,int>(it,idx)); //sorted label values
        idx++;
    }

    //weight the edge strength graph-wise
    for (size_t i = 0; i < n; i++)
    {
        double edge_w = 0;
        // sum of edge weights
        edge_w = E[i].col(2).sum();
        norm_edge.col(i) = E[i].col(2);
        norm_edge.col(i).array() /= edge_w;
    }


    //build neighborhood label vector for each vertex
    MatrixXd node_nei_vec = MatrixXd::Zero(v_all, n_label_vals);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < E[i].rows(); j++)
        {
            int edge_0 = V_label[i][E[i](j,0)];
            int edge_1 = V_label[i][E[i](j,1)];

            //add to the neighborhood list
            node_nei_vec(edge_0,label_index.at(edge_0))+=norm_edge(j,i);
            node_nei_vec(edge_1, label_index.at(edge_1)) += norm_edge(j,i);
        }
    }


    //iterate the predefined iteration and update kernel value
    K_mat.resize(n,n);
    for(int h=0;h<h_max;h++){
        //iterate all graphs
        for(int i=0;i<n;i++){
            int offset = 0;
            for (int j = i; j < n; j++)
            {
                //TODO iterate over all label value vertex! change the vertex numbering way! 
                for(auto &val:n_label){
                    inverted_index[i].find(val);
                }
                
                
            }
            
        }
    }
}
int main()
{
    MatrixXd ones = MatrixXd::Ones(20, 20);
    unordered_map<int, int> test;
    int &a = test[1];
    a=1;
    cout << test[1]<< endl;
    return 0;
}
