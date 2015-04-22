#ifndef __measure_drivers_h__
#define __measure_drivers_h__

#include <vector>
#include "data_structures.h"
#include "running_stat.h"

typedef std::vector< double (*)(const data_structures<double> &ds) > observables_vector_real;
typedef std::vector< arma::cx_double (*)(const data_structures<arma::cx_double> &ds) > observables_vector_complex;

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	const arma::vec &coefficients,
	data_structures<type> &ds,
	running_stat &E,
	gradient_running_stat &G);

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	data_structures<type> &ds,
	std::vector<running_stat>& F);

#endif

