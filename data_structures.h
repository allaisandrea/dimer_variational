#ifndef __data_structures_h__
#define __data_structures_h__

#include <armadillo>

template <class type>
struct data_structures
{
	unsigned int L;
	unsigned int Nf[2];
	unsigned int n_faces;
	arma::umat face_edges;
	arma::umat adjacent_faces;
	arma::uvec particles;
	arma::uvec edge_of[2];
	arma::Mat<type> psi[2];
	arma::Col<type> phi;
	arma::vec w[2];
	arma::uvec J[2], K[2];
	arma::Mat<type> M[2];  
	arma::Mat<type> Mi[2]; 
};

template<class type>
void build_graph(unsigned int L, data_structures<type> &ds);
#endif

