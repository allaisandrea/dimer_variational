#ifndef __linear_algebra_h__
#define __linear_algebra_h__
#include <armadillo>

template<class type>
bool singular(const arma::Mat<type> &M);


template<class type>
void rank_k_update(type a, const arma::Mat<type> &_U, const arma::Mat<type>&_V, arma::Mat<type>& _A);
#endif