#ifndef __measure_drivers_h__
#define __measure_drivers_h__

#include <vector>
#include "data_structures.h"

typedef std::vector< double (*)(const data_structures<double> &ds) > observables_vector_real;
typedef std::vector< arma::cx_double (*)(const data_structures<arma::cx_double> &ds) > observables_vector_complex;

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	data_structures<type> &ds,
	arma::Mat<type> &F,
	arma::Mat<type> &dZ);

template <class type>
void autocorrelations(const arma::Mat<type> &F, arma::mat &sF);

template <class type>
void total_energy(
	const arma::Mat<type> &_F,
	const arma::Mat<type> &dZ,
	const arma::vec & coefficients,
	double &E,
	double &sE,
	arma::vec &G,
	arma::vec &sG);

#endif

