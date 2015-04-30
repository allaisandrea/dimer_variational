#ifndef __observables_h__
#define __observables_h__

#include "data_structures.h"
#include "linear_algebra.h"
#include "utilities.h"

template<class type>
void partition_function_gradient(const data_structures<type> &ds, arma::Col<type> &G);

template<class type>
type boson_hopping(const data_structures<type> &ds);

template<class type>
type boson_potential(const data_structures<type> &ds);

template<class type>
type fermion_hopping_1(const data_structures<type> &ds);

template<class type>
type fermion_hopping_2(const data_structures<type> &ds);

template<class type>
type fermion_hopping_3(const data_structures<type> &ds);

void quasiparticle_residual(const data_structures<double> &ds, arma::mat &zo, arma::mat &ze);
#endif
