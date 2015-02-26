#ifndef __data_structures_h__
#define __data_structures_h__

#include <armadillo>

template <class type>
struct data_structures
{
	unsigned int L;
	unsigned int Nu, Nd;
	unsigned int n_faces;
	arma::umat face_edges;
	arma::umat adjacent_faces;
	arma::uvec particles;
	arma::Mat<type> psi_u, psi_d;
	arma::Col<type> phi;
	arma::vec w_u, w_d;
	arma::uvec Ju, Ku, Jd, Kd;
	arma::Mat<type> Mu; 
	arma::Mat<type> Md; 
	arma::Mat<type> Mui; 
	arma::Mat<type> Mdi;
};

template<class type>
void build_graph(unsigned int L, data_structures<type> &ds);
#endif

