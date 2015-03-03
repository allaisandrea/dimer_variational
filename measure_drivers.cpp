#include <vector>
#include "rng.h"
#include "data_structures.h"
#include "monte_carlo.h"
#include "observables.h"

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	data_structures<type> &ds,
	arma::Mat<type> &F,
	arma::Mat<type> &dZ)
{
	unsigned int i, j, n_thermalize;
	double dummy;
	arma::Col<type> buf;
	
	initial_configuration(ds);
	
	n_thermalize = round(0.05 * n_measure);
	for(i = 0; i < n_thermalize; i++)
	{
		for(j = 0; j < n_skip + 1; j++)
		{
			rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, dummy, ds);
			swap_states(0, dummy, ds);
			swap_states(1, dummy, ds);
		}
	}
	
	F.set_size(observables.size(), n_measure);
	dZ.set_size(ds.n_derivatives, n_measure);
	for(i = 0; i < n_measure; i++)
	{
		for(j = 0; j < n_skip + 1; j++)
		{
			rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, dummy, ds);
			swap_states(0, dummy, ds);
			swap_states(1, dummy, ds);
		}
		partition_function_gradient(ds, buf);
		dZ.col(i) = buf;
		for(j = 0; j < observables.size(); j++)
		{
			F(j, i) = observables[i](ds);
		}
		
	}
}

template
void monte_carlo_driver<double>(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< double (*)(const data_structures<double> &ds) > &observables,
	data_structures<double> &ds,
	arma::Mat<double> &F,
	arma::Mat<double> &dZ);

template
void monte_carlo_driver<arma::cx_double>(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< arma::cx_double (*)(const data_structures<arma::cx_double> &ds) > &observables,
	data_structures<arma::cx_double> &ds,
	arma::Mat<arma::cx_double> &F,
	arma::Mat<arma::cx_double> &dZ);