#ifndef __linear_algebra_h__
#define __linear_algebra_h__
#include <armadillo>

template<class type>
bool singular(const arma::Mat<type> &M);

void rank_1_update(double a, const arma::Col<double> &U, const arma::Row<double>&V, arma::Mat<double>& A);
void rank_1_update(arma::cx_double a, const arma::Mat<arma::cx_double> &U, const arma::Mat<arma::cx_double>&V, arma::Mat<arma::cx_double>& A);

template<class type>
void rank_k_update(type a, const arma::Mat<type> &_U, const arma::Mat<type>&_V, arma::Mat<type>& _A);

template <class type>
type trace_of_product(const arma::Mat<type> &_M1, const arma::Mat<type> &_M2);

template <class type>
void eigensystem_variation(const arma::Mat<type> U, const arma::vec w, const arma::Mat<type> V, arma::Mat<type> &dU, arma::vec &dw);

template<class type>
void copy_line(type c1, type c2, unsigned int m, unsigned int ldA, const type * A, unsigned int ldB, type* B);

template<class type>
void copy_line_sparse(type c1, type c2, unsigned int m, const unsigned int *j, unsigned int ldA, const type * A, unsigned int ldB, type* B);

#endif