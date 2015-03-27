#include <vector>
#include <ctime>
#include <iomanip>
#include "rng.h"
#include "data_structures.h"
#include "monte_carlo.h"
#include "observables.h"

template <class type>
void monte_carlo_driver(
	unsigned int n_measure,
	unsigned int n_skip,
	bool measure_gradient,
	bool shuffle_states,
	const std::vector< type (*)(const data_structures<type> &ds) > &observables,
	data_structures<type> &ds,
	arma::Mat<type> &F,
	arma::Mat<type> &dZ,
	arma::umat *J)
{
	unsigned int i, j, n_thermalize, start_time;
	double dummy;
	arma::Col<type> buf;
	
	
	start_time = std::clock();
	initial_configuration(ds);
	std::cout << "time: " << 1. * (std::clock() - start_time) / CLOCKS_PER_SEC << "\n";
	
	start_time = std::clock();
	n_thermalize = round(0.05 * n_measure);
	for(i = 0; i < n_thermalize; i++)
	{
		for(j = 0; j < n_skip + 1; j++)
		{
			rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, dummy, ds);
			if(shuffle_states)
			{
				swap_states(0, dummy, ds);
				swap_states(1, dummy, ds);
			}
		}
	}
	
	F.set_size(observables.size(), n_measure);
	if(measure_gradient)
		dZ.set_size(ds.n_derivatives, n_measure);
	J[0].set_size(ds.J[0].n_rows, n_measure);
	J[1].set_size(ds.J[1].n_rows, n_measure);
	
	for(i = 0; i < n_measure; i++)
	{
		for(j = 0; j < n_skip + 1; j++)
		{
			rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, dummy, ds);
			if(shuffle_states && j % 10 == 0)
			{
				swap_states(rng::uniform_integer(2), dummy, ds);
			}
		}
		if(measure_gradient)
		{
			partition_function_gradient(ds, buf);
			dZ.col(i) = buf;
		}
		for(j = 0; j < observables.size(); j++)
		{
			F(j, i) = observables[j](ds);
		}
		
		J[0].col(i) = ds.J[0];
		J[1].col(i) = ds.J[1];
	}
	std::cout << "time: " << 1. * (std::clock() - start_time) / CLOCKS_PER_SEC << "\n";
}

template <class type>
void autocorrelations(const arma::Mat<type> &F, arma::mat &sF)
{
	unsigned int i, j, p, n_ac = 20, n_observables, n_measure;
	double mean1, mean2;
	
	n_observables = F.n_rows;
	n_measure = F.n_cols;
	sF.set_size(n_ac, n_observables);
	for(i = 0; i < n_observables; i++)
	for(p = 0; p < n_ac; p++)
	{
		mean1 = mean2 = 0.;
		for(j = 0; j + p < n_measure; j++)
		{
			mean1 += (real(F(i, j)) - mean1) / (j + 1.);
			mean2 += (real(F(i, j + p)) - mean2) / (j + 1.);
		}
		
		sF(p, i) = 0.;
		for(j = 0; j + p < n_measure; j++)
		{
			sF(p, i) += ((real(F(i, j)) - mean1) * (real(F(i, j + p)) - mean2) - sF(p, i)) / (j + 1);
		}
	}
}

template <class type>
void total_energy(
	const arma::Mat<type> &_F,
	const arma::Mat<type> &dZ,
	const arma::vec & coefficients,
	double &E,
	double &sE,
	arma::vec &G,
	arma::vec &sG)
{
	unsigned int i, j, k, p, n_boot = 200, n_observables, n_measure, n_derivatives;
	arma::Col<type> F;
	arma::vec E_boot, dZ_boot, EdZ_boot;
	arma::mat G_boot;
	double mean1, mean2;
	
	n_observables = _F.n_rows;
	n_measure = _F.n_cols;
	n_derivatives = dZ.n_rows;
	
	F = trans(_F) * coefficients;
	E_boot.zeros(n_boot);
	if(n_derivatives > 0)
		G_boot.zeros(n_derivatives, n_boot);
	
	for(i = 0; i < n_boot; i++)
	{
		if(n_derivatives > 0)
		{
			dZ_boot.zeros(n_derivatives);
			EdZ_boot.zeros(n_derivatives);
		}
		
		for(j = 0; j < n_measure; j++)
		{
			k = rng::uniform_integer(n_measure);
			E_boot(i) += (real(F(k)) - E_boot(i)) / (j + 1.);
			if(n_derivatives > 0)
			{
				dZ_boot += (real(dZ.col(k)) - dZ_boot) / (j + 1.);
				EdZ_boot += (real(F(k) * dZ.col(k)) - EdZ_boot) / (j + 1.);
			}
		}
		
		if(n_derivatives > 0)
			G_boot.col(i) = EdZ_boot - E_boot(i) * dZ_boot;
	}
	
	E = mean(E_boot);
	sE = stddev(E_boot);
	
	if(n_derivatives > 0)
	{
		G = mean(G_boot, 1);
		sG = stddev(G_boot, 0, 1);
	}
}

template
void monte_carlo_driver<double>(
	unsigned int n_measure,
	unsigned int n_skip,
	bool shuffle_states,
	bool measure_gradient,
	const std::vector< double (*)(const data_structures<double> &ds) > &observables,
	data_structures<double> &ds,
	arma::Mat<double> &F,
	arma::Mat<double> &dZ,
	arma::umat *J);

template
void monte_carlo_driver<arma::cx_double>(
	unsigned int n_measure,
	unsigned int n_skip,
	bool shuffle_states,
	bool measure_gradient,
	const std::vector< arma::cx_double (*)(const data_structures<arma::cx_double> &ds) > &observables,
	data_structures<arma::cx_double> &ds,
	arma::Mat<arma::cx_double> &F,
	arma::Mat<arma::cx_double> &dZ,
	arma::umat *J);


template
void autocorrelations<double>(const arma::Mat<double> &F, arma::mat &sF);

template
void autocorrelations<arma::cx_double>(const arma::Mat<arma::cx_double> &F, arma::mat &sF);

template
void total_energy<double>(
	const arma::Mat<double> &_F,
	const arma::Mat<double> &dZ,
	const arma::vec & coefficients,
	double &E,
	double &sE,
	arma::vec &G,
	arma::vec &sG);

template
void total_energy<arma::cx_double>(
	const arma::Mat<arma::cx_double> &_F,
	const arma::Mat<arma::cx_double> &dZ,
	const arma::vec & coefficients,
	double &E,
	double &sE,
	arma::vec &G,
	arma::vec &sG);