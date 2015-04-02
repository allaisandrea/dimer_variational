#ifndef __minimization_h__
#define __minimization_h__
#include <armadillo>
// void simplex_minimize(arma::mat &_points, void (*f)(const arma::vec&, arma::vec&, void *), void *param);
void simplex_minimize(arma::mat &_points, void (*f)(const arma::vec&, arma::running_stat<double>&, void *), void *param);
#endif
