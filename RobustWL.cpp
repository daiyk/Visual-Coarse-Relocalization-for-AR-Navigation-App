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
// #include <opencv2/opencv.hpp>
// #include <opencv2/xfeatures2d.hpp>
using namespace std;
using namespace Eigen;

double KernelValue(const vector<int> &map1, const vector<int> &map2, int &i, int &j, vector<int> &num_v, MatrixXd &node_nei);

/**
 *E: the vector of edge matrix, edge = (E[,0], E[,1]), edge strength=E[,2]
 *V_label: label values for per-graph per node, outer length V_label must equal E outer length
 *h_max: the total steps or iterations
 *V_label: label value(int) for each graph
 *num_v: num of vertices for each graph
 *num_e: num of edge for each graph
 *K_mat: kernel matrix that stores value for graphs comparison.

**/
void WLRobustKernel(vector<MatrixXi> &E, vector<vector<int>> &V_label, vector<int> &num_v, vector<int> &num_e, int h_max, MatrixXd &K_mat)
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
        raise+=num_v[i];
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
                        K_mat(j,i) += kernelInc;                        
                    }
                }
            }
        }
    }
    //the kernel value of each graph to itself, k_mat(i,i) is counted twice, thus need to get back
    K_mat.diagonal() /=2;

    //normalize the kernel value matrix
    double sum_kernel= K_mat.diagonal().sum();

    //divide the kernel matrix with the total sum of self-kernel
    K_mat/=sum_kernel;
    
    //

    //still need to divide the graph's comparison score by the total sum of neighborhood comparisons of itself.
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

//testing function for the RobustWl kernel function
void test(){
    //build the dummy graphs for testing
    MatrixXi E1(3,3);
    vector<MatrixXi> E;

    //edge assignments
    E1(0,0)=0,E1(0,1)=3,E1(0,2)=2;
    E1(1,0)=0,E1(1,1)=1,E1(1,2)=2;
    E1(2,0)=1,E1(2,1)=2,E1(2,2)=1;

    E.push_back(E1);
    E.push_back(E1);
    
    //label assignment
    vector<int> label1{1,3,4,1};

    vector<vector<int>> label;
    label.push_back(label1);
    label.push_back(label1);

    //num of vertices
    vector<int> num_v{(int)label1.size(),(int)label1.size()};
    //numof edge
    vector<int> num_e{(int)E1.rows(),(int)E1.rows()};

    MatrixXd k_mat(2,2);

    WLRobustKernel(E,label,num_v,num_e,1,k_mat);
    cout<<k_mat<<endl;

    /**
     *E: the vector of edge matrix, edge = (E[,0], E[,1]), edge strength=E[,2]
    *V_label: label values for per-graph per node, outer length V_label must equal E outer length
    *h_max: the total steps or iterations
    *V_label: label value(int) for each graph
    *num_v: num of vertices for each graph
    *num_e: num of edge for each graph
    *K_mat: kernel matrix that stores value for graphs comparison.
    **/
    


}


int main()
{
    MatrixXd ones = MatrixXd::Ones(20, 20);
    unordered_map<int, int> testmap;
    VectorXd one1 = ones.row(5);
    int &a = testmap[1];
    a = 1;
    cout << testmap[1] << endl;
    test();
    return 0;
}
