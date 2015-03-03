#ifndef __measure_drivers_h__
#define __measure_drivers_h__

#include <vector>
#include "data_structures.h"

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	data_structures<type> &ds,
	arma::Mat<type> &F,
	arma::Mat<type> &dZ);

#endif

