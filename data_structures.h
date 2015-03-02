#ifndef __data_structures_h__
#define __data_structures_h__

#include <armadillo>
#include <set>

template <class type>
struct data_structures
{
	unsigned int L;
	unsigned int Nf[2];
	unsigned int n_faces;
	arma::umat face_edges;
	arma::umat adjacent_faces;
	arma::uvec particles;
	arma::uvec fermion_edge[2];
	std::set<unsigned int> boson_edges;
	
	arma::Mat<type> psi[2];
	arma::Cube<type> Dpsi[2];
	
	arma::Col<type> phi;
	arma::Mat<type> Dphi;
	
	arma::vec w[2];
	arma::mat Dw[2];
	
	arma::uvec J[2], K[2];
	arma::Mat<type> M[2];  
	arma::Mat<type> Mi[2]; 
};

template<class type>
void build_graph(unsigned int L, data_structures<type> &ds);
#endif

