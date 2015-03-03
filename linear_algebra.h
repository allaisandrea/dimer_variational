#ifndef __linear_algebra_h__
#define __linear_algebra_h__
#include <armadillo>

template<class type>
bool singular(const arma::Mat<type> &M);

template<class type>
void rank_k_update(type a, const arma::Mat<type> &_U, const arma::Mat<type>&_V, arma::Mat<type>& _A);

template <class type>
type trace_of_product(const arma::Mat<type> &_M1, const arma::Mat<type> &_M2);

template <class type>
void eigensystem_variation(const arma::Mat<type> U, const arma::vec w, const arma::Mat<type> V, arma::Mat<type> &dU, arma::vec &dw);
#endif