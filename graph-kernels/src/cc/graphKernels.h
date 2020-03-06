/*
Graph kernel implementation
Copyright (C) 2015 Mahito Sugiyama

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Contact:
Mahito Sugiyama
ISIR, Osaka University,
8-1, Mihogaoka, Ibaraki-shi, Osaka, 567-0047, Japan
E-mail: mahito@ar.sanken.osaka-u.ac.jp
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <set>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <igraph.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

typedef Eigen::Triplet<double> T;

using namespace std;
using namespace Eigen;

// MAIN FUNCTION
void graphKernelMatrix(vector<igraph_t>& g, vector<double>& par, string& kernel_name, MatrixXd& K);
// functions used in the main function
int kernelTable(string& kernel_name);
double computeKernelValue(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, vector<double>& par, string& kernel_name);
// igraph to Eigen
const char *getVlabel(igraph_t g, igraph_vector_t vtypes, igraph_strvector_t vnames);
const char *getElabel(igraph_t g, igraph_vector_t etypes, igraph_strvector_t enames);
void igraphToEigen(igraph_t g, MatrixXi& e, vector<int>& v_label, int *vcount, int *ecount, int *dmax);
// functions used in kernels
double selectLinearGaussian(vector<int>& h1, vector<int>& h2, double sigma);
int productMapping(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, MatrixXi& H);
void productAdjacency(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, MatrixXi& H, SparseMatrix<double>& Ax);
void bucketsort(vector<int>& x, vector<int>& index, int label_max);
// Each kernel (the full kernel matrix is computed in Weisfeiler-Lehman graph kernel)
double edgeHistogramKernel(MatrixXi& e1, MatrixXi& e2, double sigma);
double vertexHistogramKernel(vector<int>& v1_label, vector<int>& v2_label, double sigma);
double vertexEdgeHistogramKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double sigma);
double vertexVertexEdgeHistogramKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double lambda);
double geometricRandomWalkKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double lambda);
double exponentialRandomWalkKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double beta);
double kstepRandomWalkKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, vector<double>& lambda_list);
void WLKernelMatrix(vector<MatrixXi>& E, vector<vector<int> >& V_label, vector<int>& num_v, vector<int>& num_e, vector<int>& degree_max, int h_max, MatrixXd& K_mat);


// ======================================================= //
// ==================== MAIN FUNCTION ==================== //
// ======================================================= //
void graphKernelMatrix(vector<igraph_t>& g, vector<double>& par, string& kernel_name, MatrixXd& K) {
  // use attributes
  igraph_i_set_attribute_table(&igraph_cattribute_table); //experimental attribute handler that can be used from C code

  // initialization
  int n = (int)g.size();
  vector<MatrixXi> E(n);
  vector<vector<int> > V_label(n);
  vector<int> V_count(n);
  vector<int> E_count(n);
  vector<int> D_max(n);
  K.resize(n, n); //graph kernel

  // from igraph to Eigen
  for (int i = 0; i < n; i++) {
    igraphToEigen(g[i], E[i], V_label[i], &V_count[i], &E_count[i], &D_max[i]);
  }

  // compute the Kernel matrix
  if (kernelTable(kernel_name) == 11) {
    WLKernelMatrix(E, V_label, V_count, E_count, D_max, (int)par[0], K);
  } else {
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
	K(i, j) = computeKernelValue(E[i], E[j], V_label[i], V_label[j], par, kernel_name);
	K(j, i) = K(i, j);
      }
    }
  }
}


// ================================================================ //
// ==================== Functions used in MAIN ==================== //
// ================================================================ //
// A hash table for kernel names
int kernelTable(string& kernel_name) {
       if (kernel_name.compare("E") == 0) return 1;
  else if (kernel_name.compare("V")   == 0) return 2;
  else if (kernel_name.compare("VE")  == 0) return 3;
  else if (kernel_name.compare("H")   == 0) return 4;
  else if (kernel_name.compare("EG")  == 0) return 5;
  else if (kernel_name.compare("VG")  == 0) return 6;
  else if (kernel_name.compare("VEG") == 0) return 7;
  else if (kernel_name.compare("GR")  == 0) return 8;
  else if (kernel_name.compare("ER")  == 0) return 9;
  else if (kernel_name.compare("kR")  == 0) return 10;
  else if (kernel_name.compare("WL")  == 0) return 11;
  else return 1;
}
// compute a kernel value of a pair of graphs
double computeKernelValue(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, vector<double>& par, string& kernel_name) {
  double K;
  switch (kernelTable(kernel_name)) {
  case 1: // edge histogram kernel
    K = edgeHistogramKernel(e1, e2, -1.0);
    break;
  case 2: // vertex histogram kernel
    K = vertexHistogramKernel(v1_label, v2_label, -1.0);
    break;
  case 3: // vertex-edge histogram kernel
    K = vertexEdgeHistogramKernel(e1, e2, v1_label, v2_label, -1.0);
    break;
  case 4: // vertex-vertex-edge histogram kernel
    K = vertexVertexEdgeHistogramKernel(e1, e2, v1_label, v2_label, par[0]);
    break;
  case 5: // edge histogram kernel (Gaussian)
    K = edgeHistogramKernel(e1, e2, par[0]);
    break;
  case 6: // vertex histogram kernel (Gaussian)
    K = vertexHistogramKernel(v1_label, v2_label, par[0]);
    break;
  case 7: // vertex-edge histogram kernel (Gaussian)
    K = vertexEdgeHistogramKernel(e1, e2, v1_label, v2_label, par[0]);
    break;
  case 8: // geometric random walk kernel
    K = geometricRandomWalkKernel(e1, e2, v1_label, v2_label, par[0]);
    break;
  case 9: // exponential random walk kernel
    K = exponentialRandomWalkKernel(e1, e2, v1_label, v2_label, par[0]);
    break;
  case 10: // k-step random walk kernel
    K = kstepRandomWalkKernel(e1, e2, v1_label, v2_label, par);
    break;
  default:
    K = 0;
    break;
  }
  return K;
}


// ========================================================= //
// ==================== igraph to Eigen ==================== //
// ========================================================= //
// check and get vertex labels
const char *getVlabel(igraph_t g, igraph_vector_t vtypes, igraph_strvector_t vnames) {
  //here we assume there is only one attributes' name of vertex except the "id" attribute
  const char *vlabel;
  if (igraph_strvector_size(&vnames) > 2) {
    cerr << "ERROR: There are multiple types of vertex labels." << endl;
    exit(1);
  } else if (igraph_strvector_size(&vnames) == 2) {
    for (int i = 0; i < igraph_strvector_size(&vnames); i++) {
      if (strcmp(STR(vnames, i), "id") != 0) { //as long as it doesn't equals to "id" then record it
	vlabel = STR(vnames, i); //put the second attri of labels to the vlabel

	if (VECTOR(vtypes)[i] != IGRAPH_ATTRIBUTE_NUMERIC) {
	  cerr << "ERROR: Vertex attributes should be numeric values." << endl;
	  exit(1);
	}
	for (int j = 0; j < igraph_vcount(&g); j++) {
	  if (isnan(VAN(&g, vlabel, j))) {
	    cerr << "ERROR: Attribute of the vertex " << j << " is missing." << endl;
	    exit(1);
	  }
	}
      }
    }
  } else {
    vlabel = NULL;
  }
  return vlabel;
}
// check and get edge labels
const char *getElabel(igraph_t g, igraph_vector_t etypes, igraph_strvector_t enames) {
  const char *elabel;
  if (igraph_strvector_size(&enames) > 1) {
    cerr << "ERROR: There are multiple types of edge labels." << endl;
    exit(1);
  } else if (igraph_strvector_size(&enames) == 1) {
    elabel = STR(enames, 0);
    if (VECTOR(etypes)[0] != IGRAPH_ATTRIBUTE_NUMERIC) {
      cerr << "ERROR: Edge attributes should be numeric values." << endl;
      exit(1);
    }
    for (int j = 0; j < igraph_ecount(&g); j++) {
      if (isnan(EAN(&g, STR(enames, 0), j))) {
	cerr << "ERROR: Attribute of the vertex " << j << " is missing." << endl;
	exit(1);
      }
    }
  } else {
    elabel = NULL;
  }
  return elabel;
}
// wrapper function from igraph to Eigen
void igraphToEigen(igraph_t g, MatrixXi& e, vector<int>& v_label, int *vcount, int *ecount, int *dmax) {
  // initialization, e is the MatrixXi for assignment
  igraph_vector_t edge, gtypes, vtypes, etypes;
  igraph_strvector_t gnames, vnames, enames;
  igraph_vector_init(&edge, 0);
  igraph_vector_init(&gtypes, 0);
  igraph_vector_init(&vtypes, 0);
  igraph_vector_init(&etypes, 0);
  igraph_strvector_init(&gnames, 0); //Reserves memory for the string vector, a string vector must be first initialized before calling other functions on it. All elements of the string vector are set to the empty string.
  igraph_strvector_init(&vnames, 0);
  igraph_strvector_init(&enames, 0);
  // resize
  e.resize(igraph_ecount(&g), 3); //resize to (n_edges,3)

  // Number of vertices
  *vcount = igraph_vcount(&g); //num of vertex
  // Number of edges
  *ecount = igraph_ecount(&g); //num of edges
  // Maximum degree
  igraph_maxdegree(&g, dmax, igraph_vss_all(), IGRAPH_ALL, 0); // calculate the maximum degree inside a graph

  // get attributes
  igraph_cattribute_list(&g, &gnames, &gtypes, &vnames, &vtypes, &enames, &etypes);
  // check and get vertex labels
  const char *vlabel;
  vlabel = getVlabel(g, vtypes, vnames); // label represents the attributes name
  // check edge labels
  const char *elabel;
  elabel = getElabel(g, etypes, enames); // Elabel represents the attributes name

  // copy from igraph to Eigen
  // - edge
  igraph_get_edgelist(&g, &edge, 0);//get the edge list and store it in the order of ids.
  for (int i = 0; i < igraph_ecount(&g); i++) {
    e(i, 0) = (int)VECTOR(edge)[i * 2];
    e(i, 1) = (int)VECTOR(edge)[i * 2 + 1];
    e(i, 2) = elabel != NULL ? (int)EAN(&g, elabel, i) : 1; // check whether elabel exist
  }
  // if (elabel == NULL) cout << "Edge labels are missing, \"1\" is inserted to every edge." << endl;
  // - vertex
  if (vlabel != NULL) {
    for (int i = 0; i < igraph_vcount(&g); i++) {
      v_label.push_back((int)VAN(&g, vlabel, i)); // push back edge label value
    }
  } else {
    for (int i = 0; i < igraph_vcount(&g); i++) {
      v_label.push_back(1);
    }
  }
  // if (vlabel == NULL) cout << "Vertex labels are missing, \"1\" is inserted to every vertex." << endl;

  igraph_vector_destroy(&edge);
  igraph_vector_destroy(&gtypes);
  igraph_vector_destroy(&vtypes);
  igraph_vector_destroy(&etypes);
  igraph_strvector_destroy(&gnames);
  igraph_strvector_destroy(&vnames);
  igraph_strvector_destroy(&enames);
}


// ===================================================== //
// ==================== Each kernel ==================== //
// ===================================================== //
// edge histogram karnel
double edgeHistogramKernel(MatrixXi& e1, MatrixXi& e2, double sigma) {
  int e_label_max = 0;
  for (int i = 0; i < e1.rows(); i++) {
    if (e1(i, 2) > e_label_max) e_label_max = e1(i, 2);
  }
  for (int i = 0; i < e2.rows(); i++) {
    if (e2(i, 2) > e_label_max) e_label_max = e2(i, 2);
  }

  vector<int> h1(e_label_max + 1, 0);
  vector<int> h2(e_label_max + 1, 0);

  for (int i = 0; i < e1.rows(); i++) {
    (h1[e1(i, 2)])++;
  }
  for (int i = 0; i < e2.rows(); i++) {
    (h2[e2(i, 2)])++;
  }

  return selectLinearGaussian(h1, h2, sigma);
}
// vertex histogram karnel
double vertexHistogramKernel(vector<int>& v1_label, vector<int>& v2_label, double sigma) {
  int v1_label_max = *max_element(v1_label.begin(), v1_label.end());
  int v2_label_max = *max_element(v2_label.begin(), v2_label.end());
  int v_label_max = v1_label_max > v2_label_max ? v1_label_max : v2_label_max;

  vector<int> h1(v_label_max + 1, 0);
  vector<int> h2(v_label_max + 1, 0);

  for (int i = 0; i < (int)v1_label.size(); i++) {
    (h1[v1_label[i]])++;
  }
  for (int i = 0; i < (int)v2_label.size(); i++) {
    (h2[v2_label[i]])++;
  }

  return selectLinearGaussian(h1, h2, sigma);
}
// vertex-edge histogram karnel
double vertexEdgeHistogramKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double sigma) {
  int e_label_max = 0;
  for (int i = 0; i < e1.rows(); i++) {
    if (e1(i, 2) > e_label_max) e_label_max = e1(i, 2);
  }
  for (int i = 0; i < e2.rows(); i++) {
    if (e2(i, 2) > e_label_max) e_label_max = e2(i, 2);
  }
  e_label_max++;

  int v1_label_max = *max_element(v1_label.begin(), v1_label.end());
  int v2_label_max = *max_element(v2_label.begin(), v2_label.end());
  int v_label_max = v1_label_max > v2_label_max ? v1_label_max : v2_label_max;
  v_label_max++;

  vector<int> h1(v_label_max * v_label_max * e_label_max, 0);
  vector<int> h2(v_label_max * v_label_max * e_label_max, 0);

  int v1, v2;
  for (int i = 0; i < e1.rows(); i++) {
    v1 = e1(i, 0);
    v2 = e1(i, 1);
    if (v2 > v1) { int v_tmp = v1; v1 = v2; v2 = v_tmp; }
    (h1[v1_label[v1] + v1_label[v2] * v_label_max + e1(i, 2) * v_label_max * v_label_max])++;
  }
  for (int i = 0; i < e2.rows(); i++) {
    v1 = e2(i, 0);
    v2 = e2(i, 1);
    if (v2 > v1) { int v_tmp = v1; v1 = v2; v2 = v_tmp; }
    (h2[v2_label[v1] + v2_label[v2] * v_label_max + e2(i, 2) * v_label_max * v_label_max])++;
  }

  return selectLinearGaussian(h1, h2, sigma);
}
// vertex-vertex-edge histogram karnel
double vertexVertexEdgeHistogramKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double lambda) {
  return vertexHistogramKernel(v1_label, v2_label, -1.0) + lambda * vertexEdgeHistogramKernel(e1, e2, v1_label, v2_label, -1.0);
}
// geometric random walk karnel
double geometricRandomWalkKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double lambda) {
  // map each product (v_1, v_2) of vertics to a number H(v_1, v_2)
  MatrixXi H(v1_label.size(), v2_label.size());  
  int n_vx = productMapping(e1, e2, v1_label, v2_label, H);

  // prepare identity matrix
  SparseMatrix<double> I(n_vx, n_vx);
  I.setIdentity();

  // compute the adjacency matrix Ax of the direct product graph
  SparseMatrix<double> Ax(n_vx, n_vx);
  productAdjacency(e1, e2, v1_label, v2_label, H, Ax);

  // inverse of I - lambda * Ax by fixed-point iterations
  VectorXd I_vec(n_vx);
  for (int i  = 0; i < n_vx; i++) I_vec[i] = 1;
  VectorXd x = I_vec;
  VectorXd x_pre(n_vx); x_pre.setZero();

  double eps = pow(10, -10);
  int count = 0;
  while ((x - x_pre).squaredNorm() > eps) {
    if (count > 100) {
      // cout << "does not converge until " << count - 1 << " iterations" << endl;
      break;
    }
    x_pre = x;
    x = I_vec + lambda * Ax * x_pre;
    count++;
  }
  return x.sum();
}
// exponential random walk karnel
double exponentialRandomWalkKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, double beta) {
  // map each product (v_1, v_2) of vertics to a number H(v_1, v_2)
  MatrixXi H(v1_label.size(), v2_label.size());  
  int n_vx = productMapping(e1, e2, v1_label, v2_label, H);

  // compute the adjacency matrix Ax of the direct product graph
  SparseMatrix<double> Ax(n_vx, n_vx);
  productAdjacency(e1, e2, v1_label, v2_label, H, Ax);

  // compute e^{beta * Ax}
  SelfAdjointEigenSolver<MatrixXd> es(Ax);
  VectorXd x = (beta * es.eigenvalues()).array().exp();
  MatrixXd D = x.asDiagonal();  
  MatrixXd V = es.eigenvectors();

  MatrixXd I(n_vx, n_vx);
  I.setIdentity();
  FullPivLU<MatrixXd> solver(V);
  MatrixXd V_inv = solver.solve(I);
  MatrixXd Res = V * D * V_inv;

  // compute the total sum
  double K = 0;
  for (int i = 0; i < Res.rows(); i++) {
    for (int j = 0; j < Res.cols(); j++) {
      K += Res(i, j);
    }
  }

  return K;
}
// k-step product graph karnel
double kstepRandomWalkKernel(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, vector<double>& lambda_list) {
  // map each product (v_1, v_2) of vertics to a number H(v_1, v_2)
  MatrixXi H(v1_label.size(), v2_label.size());
  int n_vx = productMapping(e1, e2, v1_label, v2_label, H);

  // prepare identity matrix
  SparseMatrix<double> I(n_vx, n_vx);
  I.setIdentity();

  // compute the adjacency matrix Ax of the direct product graph
  SparseMatrix<double> Ax(n_vx, n_vx);
  productAdjacency(e1, e2, v1_label, v2_label, H, Ax);

  // compute products until k
  int k_max = (int)lambda_list.size() - 1;
  SparseMatrix<double> Ax_pow = I;
  SparseMatrix<double> Sum = lambda_list[0] * I;
  for (int k = 1; k <= k_max; k++) {
    Ax_pow = Ax * Ax_pow;
    Sum += lambda_list[k] * Ax_pow;
  }

  // compute the total sum
  double K = 0;
  for (int i = 0; i < Sum.outerSize(); ++i) {
    for (SparseMatrix<double>::InnerIterator it(Sum, i); it; ++it) {
      K += it.value();
    }
  }

  return K;
}
// Weisfeiler-Leiman graph kernel
void WLKernelMatrix(vector<MatrixXi>& E, vector<vector<int> >& V_label, vector<int>& num_v, vector<int>& num_e, vector<int>& degree_max, int h_max, MatrixXd& K_mat) {
  //E is the edge matrix
  K_mat.setZero();
  int n = (int)E.size(); // num of graphs
  int v_all = accumulate(num_v.begin(), num_v.end(), 0); //compute the sum from begin to end, and add to 0. The num of vertices forall graphs
  int degree_max_all = *max_element(degree_max.begin(), degree_max.end()); //get the max degree for all these graphs
  vector<int> label_max_vec(n); // max vertex attri values for each graph 
  for (int i = 0; i < n; i++) {
    label_max_vec[i] = *max_element(V_label[i].begin(), V_label[i].end()); //compute label with maximum value for each graphs
  }
  int label_max = *max_element(label_max_vec.begin(), label_max_vec.end()); // label with maximum value over all graphs

  int raise = 0;
  vector<int> counter(*max_element(num_v.begin(), num_v.end())); // num of vertex of the maximum graph
  MatrixXi nei_list(v_all, degree_max_all + 1);// shape=[total vertex, maximum degree of graph]
  MatrixXi label_list(v_all, h_max + 1); // h_max is the parameters[0] for ?
  vector<int> x(v_all); // create vector with length total vertex
  vector<int> index(v_all); // create vector with total length vertex
  vector<int> index_org(v_all);// create vector with total length vertex
  vector<int> graph_index(v_all);// create vector with total length vertex

  label_list.setZero(); //label list records every iteration's label value

  for (int i = 0; i < n; i++) { //n: num of graphs
    for (int j = 0; j < num_v[i]; j++) {// num_v: num of vertex for each graph
      label_list(j + raise, 0) = V_label[i][j]; // feed the eigen matrix with label values, 2d matrix with content 
      graph_index[j + raise] = i; // each index value index the no. of graphs
    }
    raise += num_v[i];// offset for each graph
  }


  // ===== Increment kernel values using the initial vertex labels =====
  // radix sort
  for (int i = 0; i < v_all; i++) {
    index[i] = i;
    index_org[i] = i;
    x[i] = label_list(i, 0);// feed x with label value
  }
  bucketsort(x, index, label_max); // get the "index", that sort according to the label values 

  // add kernel values
  vector<int> count(n);
  set<int> count_index;
  int k_value;
  for (int i = 0; i < v_all; i++) {
    //label value from small to large
    count_index.insert(graph_index[index_org[index[i]]]); // index[i]: index value, sorted index value at label value's rank-i. index_org[]: index it self. count_index: for each vertex, show its original graph's no, according to sorted label value  
    count[graph_index[index_org[index[i]]]]++; // count the number of vertex in each graph
    if (i == v_all - 1 || label_list(index[i], 0) != label_list(index[i + 1], 0)) { //when there is a jump in label values (or final value), which means the current stage all same value's label has been accumulated.
      for (set<int>::iterator itr = count_index.begin(), end = count_index.end(); itr != end; ++itr) {
        for (set<int>::iterator itr2 = itr, end2 = count_index.end(); itr2 != end2; ++itr2) {
          k_value = count[*itr] * count[*itr2]; // compute the similarity for this round's label value, compute any similar value number's square. count^2 
          K_mat(*itr, *itr2) += k_value;
          K_mat(*itr2, *itr) += k_value;
        }
        count[*itr] = 0; // reset
      }
      count_index.clear(); // reset
    }
  }

  int v_raised_1, v_raised_2;
  for (int h = 0; h < h_max; h++) {
    nei_list.setZero(); // shape=[total vertex, maximum degree of graph] adjacency list?

    // first put vertex label
    nei_list.col(0) = label_list.col(h); // we only know that label_list[:,0] is the label value, label_list[:,1:h] are zeros or label for the next iterations
    // second put neighbor labels
    raise = 0;
    for (int i = 0; i < n; i++) {
      fill(counter.begin(), counter.end(), 1); // counter's size = the maximum size of vertex for these graph: 1:vertex_size
      for (int j = 0; j < num_e[i]; j++) {
        v_raised_1 = E[i](j, 0) + raise; // find the corresponding edge in graph i
        v_raised_2 = E[i](j, 1) + raise;
        nei_list(v_raised_1, counter[E[i](j, 0)]) = label_list(v_raised_2, h); // 将上一轮循环的label_value赋予nei_list 
        nei_list(v_raised_2, counter[E[i](j, 1)]) = label_list(v_raised_1, h); // put edge's other end's label value on [node,1]
        counter[E[i](j, 0)]++; // the degree at this node increment, 只记录label value
        counter[E[i](j, 1)]++;
      }
      raise += num_v[i]; // index offset over graph
    }

    // sort each row w.r.t. neighbors
    vector<int> y(nei_list.cols() - 1); // y.size=cols-1
    for (int i = 0; i < v_all; i++) {
      for (int j = 1; j < nei_list.cols(); ++j) {
	      y[j - 1] = nei_list(i, j); // store all neighbor node to y (except the zero)
      }
      sort(y.begin(), y.end(), greater<int>());
      for (int j = 1; j < nei_list.cols(); ++j) {
	      nei_list(i, j) = y[j - 1]; // sort according to the label value and return back
      }
    }
    // radix sort
    for (int i = 0; i < v_all; i++) {
      index[i] = i;
      index_org[i] = i;
    }
    for (int k = nei_list.cols() - 1; k >= 0; k--) {
      for (int i = 0; i < v_all; i++) {
	      x[i] = nei_list(i, k); // x[i] now is the kth neighbor of all vertices
      }
      bucketsort(x, index, label_max);
    }
    // re-labeling and increment kernel values
    label_max++; // plus 1 so that the new label start from the maxlabel+1, doesn't overlap with the existent label value
    for (int i = 0; i < v_all; i++) {
      label_list(index_org[index[i]], h + 1) = label_max; // next round's label start from the label_max,  will give label_list the maximum label value 
      count_index.insert(graph_index[index_org[index[i]]]);
      count[graph_index[index_org[index[i]]]]++;
      if (i == v_all - 1 ||
	  (nei_list.row(index[i]) - nei_list.row(index[i + 1])).array().abs().sum() != 0) {
	for (set<int>::iterator itr = count_index.begin(), end = count_index.end(); itr != end; ++itr) {
	  for (set<int>::iterator itr2 = itr, end2 = count_index.end(); itr2 != end2; ++itr2) {
	    k_value = count[*itr] * count[*itr2];
	    K_mat(*itr, *itr2) += k_value;
	    K_mat(*itr2, *itr) += k_value;
	  }
	  count[*itr] = 0;
	}
	count_index.clear();	
	label_max++;
      }
    }
  }
  K_mat.diagonal() /= 2;
}


// =================================================================== //
// ==================== Functions used in kernels ==================== //
// =================================================================== //
// select linear kernel or Gaussian kernel in histogram kernels
double selectLinearGaussian(vector<int>& h1, vector<int>& h2, double sigma) {
  double K = 0;
  if (sigma < 0) {
    // linear kernel
    for (int i = 0; i < (int)h1.size(); i++) {
      K += (double)h1[i] * (double)h2[i];
    }
  } else {
    // Gaussian kernel
    for (int i = 0; i < (int)h1.size(); i++) {
      K += ((double)h1[i] - (double)h2[i]) * ((double)h1[i] - (double)h2[i]);
    }
    K = exp(-1.0 * K / (2.0 * sigma * sigma));
  }
  return K;
}
// map each product (v_1, v_2) of vertics to a number H(v_1, v_2)
int productMapping(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, MatrixXi& H) {
  int n_vx = 0;
  for (int i = 0; i < (int)v1_label.size(); i++) {
    for (int j = 0; j < (int)v2_label.size(); j++) {
      if (v1_label[i] == v2_label[j]) {
	H(i, j) = n_vx;
	n_vx++;
      }
    }
  }
  return n_vx;
}
// compute the adjacency matrix Ax of the direct product graph (sparse)
void productAdjacency(MatrixXi& e1, MatrixXi& e2, vector<int>& v1_label, vector<int>& v2_label, MatrixXi& H, SparseMatrix<double>& Ax) {
  vector<T> v;
  for (int i = 0; i < e1.rows(); i++) {
    for (int j = 0; j < e2.rows(); j++) {      
      if (   v1_label[e1(i, 0)] == v2_label[e2(j, 0)]
	  && v1_label[e1(i, 1)] == v2_label[e2(j, 1)]
	  && e1(i, 2) == e2(j, 2)) {
	v.push_back(T(H(e1(i, 0), e2(j, 0)), H(e1(i, 1), e2(j, 1)), 1.0));
	v.push_back(T(H(e1(i, 1), e2(j, 1)), H(e1(i, 0), e2(j, 0)), 1.0));
      }
      if (   v1_label[e1(i, 0)] == v2_label[e2(j, 1)]
	  && v1_label[e1(i, 1)] == v2_label[e2(j, 0)]
	  && e1(i, 2) == e2(j, 2)) {
	v.push_back(T(H(e1(i, 0), e2(j, 1)), H(e1(i, 1), e2(j, 0)), 1.0));
	v.push_back(T(H(e1(i, 1), e2(j, 0)), H(e1(i, 0), e2(j, 1)), 1.0));
      }
    }
  }
  Ax.setFromTriplets(v.begin(), v.end());
}

// bucket sort used in Weisfeiler-Leiman graph kernel
void bucketsort(vector<int>& x, vector<int>& index, int label_max) {
  // x is the vector fed with label value, index is the index from 0 to the vertex num.
  /** for the second neighborhood sorting:
   *  first round index-> index value with cols-1 neighboor's label value
   *  
   * 
  **/
  vector<vector<int> > buckets;
  buckets.resize(label_max + 1); // resize to the maximum label value, lenth

  for (vector<int>::iterator itr = index.begin(), end = index.end(); itr != end; ++itr) {
    buckets[ x[*itr] ].push_back(*itr); // index (*itr)'s kth edge label value is x[*iter] 
  }

  int counter = 0;
  for (vector<vector<int> >::iterator itr = buckets.begin(), end = buckets.end(); itr != end; ++itr) {
    for (vector<int>::iterator itr2 = (*itr).begin(), end2 = (*itr).end(); itr2 != end2; ++itr2) {
      index[counter] = *itr2; // *iter2 is the index, 
      counter++;
    }
  } // range the vertex index according to the label value from small to large 
}
