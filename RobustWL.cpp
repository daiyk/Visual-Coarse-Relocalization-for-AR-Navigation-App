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
double KernelValue(const vector<int> &map1, const vector<int> &map2, int &i, int &j, vector<int> &num_v, MatrixXd &node_nei);

void WLRobustKernel(vector<MatrixXi> &E, vector<vector<int>> &V_label, vector<int> &num_v, vector<int> &num_e, int h_max, MatrixXd &K_mat)
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
        norm_edge.col(i) = E[i].col(2);
        norm_edge.col(i).array() /= edge_w;
    }

    //build neighborhood label vector for each vertex
    MatrixXd node_nei_vec = MatrixXd::Zero(v_all, n_label_vals);
    int raise = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < E[i].rows(); j++)
        {
            int edge_0 = V_label[i][E[i](j, 0)];
            int edge_1 = V_label[i][E[i](j, 1)];

            //add to the neighborhood list
            //need to consider the offset for different graph
            //label_index returns the label value's index
            node_nei_vec(edge_0 + raise, label_index.at(edge_0)) += norm_edge(j, i);
            node_nei_vec(edge_1 + raise, label_index.at(edge_1)) += norm_edge(j, i);
        }
        raise+=num_v[i];
    }

    //iterate the predefined iteration and update kernel value
    K_mat.resize(n, n), K_mat.setZero();
    for (int h = 0; h < h_max; h++)
    {
        //iterate all graphs
        for (int i = 0; i < n; i++)
        {
            int offset = 0;
            for (int j = i; j < n; j++)
            {
                auto &inv_vet1 = inverted_index[i];
                auto &inv_vet2 = inverted_index[j];
                //TODO iterate over all label value vertex! change the vertex numbering way!
                for (auto &val : n_label)
                {
                    //only increment the kernel value when same label is found in both graphs
                    if(inv_vet1.count(val) && inv_vet2.count(val)){
                        auto &map1 = inv_vet1[val];
                        auto &map2 = inv_vet2[val];
                        double kernelInc = KernelValue(map1,map2,i,j,num_v,node_nei_vec);
                        K_mat(i,j) += kernelInc;                        
                    }
                    

                }
            }
        }
    }
}

double KernelValue(const vector<int> &map1, const vector<int> &map2, int &i, int &j, vector<int> &num_v, MatrixXd &node_nei)
{
    //calculate the kernel value increments
    vector<double> values;
    int start1 = 0, start2=0;
    for(int m=0;m<i;m++){
        start1+=num_v[m];
    }
    for(int m=0;m<j;m++){
        start2+=num_v[m];
    }

    
    for(auto v1 : map1){
        for(auto v2 : map2){
            VectorXd label_vec1 = node_nei.row(start1+v1);
            VectorXd label_vec2 = node_nei.row(start2+v2);
            //compute kernel value
            values.push_back(label_vec1.dot(label_vec2));
        }
    }

    //return the largest dot product value
    return *max_element(values.begin(), values.end());
}

int main()
{
    MatrixXd ones = MatrixXd::Ones(20, 20);
    unordered_map<int, int> test;
    VectorXd one1 = ones.row(5);
    int &a = test[1];
    a = 1;
    cout << test[1] << endl;
    return 0;
}
