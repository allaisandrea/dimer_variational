#include "measure_drivers.h"
#include <iomanip>
#include "rng.h"
#include "data_structures.h"
#include "monte_carlo.h"
#include "observables.h"
#include "utilities.h"

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	const arma::vec &coefficients,
	data_structures<type> &ds,
	running_stat<double> &E,
	gradient_running_stat &G)
{
	unsigned int i, ii, j, n_thermalize, di_print;
	double dummy;
	arma::Col<type> dZ;
	type F;
	
	initial_configuration(ds);
	
	n_thermalize = round(0.05 * n_measure);
	
	for(i = 0; i < n_measure + n_thermalize; i++)
	{
		for(j = 0; j < n_skip + 1; j++)
			monte_carlo_step(true, j % 4 == 0, dummy, ds);
		
		if(i >= n_thermalize)
		{
			ii = i - n_thermalize;
			
			F = 0.;
			for(j = 0; j < observables.size(); j++)
			{
				F += coefficients(j) * observables[j](ds);
			}
			E(F);
			if(measure_gradient)
			{
				partition_function_gradient(ds, dZ);
				G(F, dZ);
			}
			
			di_print = n_measure / 100;
			if(di_print == 0) di_print = 1;
			if((ii + 1) % di_print == 0)
				std::cout << elapsed_time_string() << ":" << std::setw(5) << round(100. * (ii + 1) / n_measure) << " %" << std::endl;
		}
		else
		{
			di_print = n_thermalize / 5;
			if(di_print == 0) di_print = 1;
			if((i + 1) % di_print == 0)
				std::cout << elapsed_time_string() << ":" << std::setw(5) << round(100. * (i + 1) / n_thermalize) << " %" << std::endl;
		}
	}
	
}

template
void monte_carlo_driver<double>(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	const std::vector< double (*)(const data_structures<double> &ds) > &observables,
	const arma::vec &coefficients,
	data_structures<double> &ds,
	running_stat <double> &E,
	gradient_running_stat &G);


template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	data_structures<type> &ds,
	std::vector<running_stat<double> >& F)
{
	unsigned int i, ii, j, n_thermalize;
	double dummy;
	
	initial_configuration(ds);
	if(F.size() != observables.size())
	{
		F.resize(observables.size());
	}
	n_thermalize = round(0.05 * n_measure);
	for(i = 0; i < n_measure + n_thermalize; i++)
	{
		for(j = 0; j < n_skip + 1; j++)
			monte_carlo_step(true, j % 4 == 0, dummy, ds);
		
		if(i >= n_thermalize)
		{
			for(j = 0; j < observables.size(); j++)
			{
				F[j](real(observables[j](ds)));
			}
		}
	}
}

template
void monte_carlo_driver<double>(
	unsigned int n_measure,
	unsigned int n_skip,
	const std::vector< double (*)(const data_structures<double> &ds) > &observables,
	data_structures<double> &ds,
	std::vector< running_stat<double> > &F);