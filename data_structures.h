#ifndef __data_structures_h__
#define __data_structures_h__

#include <armadillo>
#include <set>
#include <exception>

template <class type>
struct data_structures
{
	unsigned int L;
	unsigned int Nf[2];
	unsigned int n_faces;
	unsigned int n_derivatives;
	
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
	
	bool is_empty(unsigned int p) const{  return p == 0;}
	bool is_boson(unsigned int p) const{  return p == 1;}
	bool is_fermion(unsigned int p) const{return p >= 2 && p < Nf[0] + Nf[1] + 2;}
	bool is_up(unsigned int p) const{     return p >= 2 && p < Nf[0] + 2;}
	bool is_dn(unsigned int p) const{     return p >= Nf[0] + 2 && p < Nf[0] + Nf[1] + 2;}
	bool is_valid(unsigned int p) const{  return p < Nf[0] + Nf[1] + 2;}
	void to_p_s(unsigned int p0, unsigned int &p, unsigned int &s)  const
	{
		if(is_up(p0))
		{
			p = p0 - 2;
			s = 1;
		}
		else if(is_dn(p0))
		{
			p = p0 - 2 - Nf[0];
			s = 0;
		}
		else
			throw std::logic_error("Particle is not a fermion");
	}
};

template<class type>
void build_graph(data_structures<type> &ds);
#endif

