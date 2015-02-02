#ifndef __data_structures_h__
#define __data_structures_h__

template <class type>
struct data_structures
{
	unsigned int Nf;
	arma::umat FE;
	arma::umat AF;
	arma::uvec P;
	arma::Mat<type> psi;
	arma::Vec<type> phi;
	arma::mat w;
	arma::uvec J;
	arma::Mat<type> Mu; 
	arma::Mat<type> Md; 
	arma::Mat<type> Mui; 
	arma::Mat<type> Mdi;
}

#endif
