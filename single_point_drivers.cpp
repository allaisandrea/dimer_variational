#include "single_point_drivers.h"
#include "measure_drivers.h"
#include "states.h"
#include "observables.h"
#include "rng.h"
#include "monte_carlo.h"

void single_point_driver(interface_2 &p)
{
	unsigned int n_observables, i;
	data_structures<double> ds;
	observables_vector_real observables;
	std::vector< running_stat> F;
	arma::vec u, u0;
	arma::mat P;
	
	std::cout << asctime(localtime(&start_time)) << std::endl;
	
	ds.L = p.L;
	ds.Nf[0] = p.Nu;
	ds.Nf[1] = p.Nd;
	
	for(i = 0; i < p.observables.n_elem; i++)
	{
		if(p.observables(i) == 0)
			observables.push_back(&boson_hopping);
		else if(p.observables(i) == 1)
			observables.push_back(&boson_potential);
		else if(p.observables(i) == 2)
			observables.push_back(&fermion_hopping_1);
		else if(p.observables(i) == 3)
			observables.push_back(&fermion_hopping_2);
		else if(p.observables(i) == 4)
			observables.push_back(&fermion_hopping_3);
		else
			throw std::logic_error("Unknown observable");
	}
	n_observables = observables.size();
	
	build_graph(ds);

	rng::seed(p.seed);
	
	P.eye(6, 6);
	u0.zeros(6);
	u << p.u0 << p.u1x << p.u1y << p.u2 << p.u3x << p.u3y;
	homogeneous_state(u, P, u0, p.beta, ds);

	F.resize(n_observables);
	for(i = 0; i < n_observables; i++)
		F[i].reset(20);
	
	monte_carlo_driver(p.n_measure, p.n_skip, observables, ds, F);
	
	p.F.set_size(2, n_observables);
	p.autocorrelation.set_size(20, n_observables);
	for(i = 0; i < n_observables; i++)
	{
		p.F(0, i) = F[i].mean();
		p.F(1, i) = sqrt(F[i].variance_of_the_mean());
		p.autocorrelation.col(i) = F[i].autocorrelation();
	}
	p.write();
}

void qp_residual_driver(interface_3 &p)
{
	unsigned int i, ii, j, n_thermalize, di_print;
	double dummy;
	data_structures <double> ds;
	arma::mat zo, ze, P;
	arma::vec u, u0;
	
	std::cout << asctime(localtime(&start_time)) << std::endl;
	
	ds.L = p.L;
	ds.Nf[0] = p.Nu;
	ds.Nf[1] = p.Nd;
	
	build_graph(ds);
	
	P.eye(6, 6);
	u0.zeros(6);
	u << p.u0 << p.u1x << p.u1y << p.u2 << p.u3x << p.u3y;
	homogeneous_state(u, P, u0, p.beta, ds);
	
	initial_configuration(ds);
	
	p.J = ds.J[0];
	p.k = ds.momenta;
	p.zo.zeros(5, ds.Nf[0]);
	p.szo.zeros(5, ds.Nf[0]);
// 	p.ze.zeros(ds.n_edges, ds.n_edges);
// 	p.sze.zeros(ds.n_edges, ds.n_edges);
	
	n_thermalize = round(0.05 * p.n_measure);
	for(i = 0; i < p.n_measure + n_thermalize; i++)
	{
		for(j = 0; j < p.n_skip + 1; j++)
			monte_carlo_step(true, false, dummy, ds);
		
		if(i >= n_thermalize)
		{
			ii = i - n_thermalize;
			
			quasiparticle_residual(ds, zo, ze);
			
			zo = (zo - p.zo) / (i + 1.);
			p.zo += zo;
			p.szo += i * zo % zo - p.szo / (i + 1.);
// 			ze = (ze - p.ze) / (i + 1.);
// 			p.ze += ze;
// 			p.sze += i * ze % ze - p.sze / (i + 1.);
			
			di_print = p.n_measure / 100;
			if(di_print == 0) di_print = 1;
			if((ii + 1) % di_print == 0)
				std::cout << elapsed_time_string() << ":" << std::setw(5) << round(100. * (ii + 1) / p.n_measure) << " %" << std::endl;
		}
		else
		{
			di_print = n_thermalize / 5;
			if(di_print == 0) di_print = 1;
			if((i + 1) % di_print == 0)
				std::cout << elapsed_time_string() << ":" << std::setw(5) << round(100. * (i + 1) / n_thermalize) << " %" << std::endl;
		}
	}
	
	p.szo = sqrt(p.szo / (p.n_measure - 1.));	
	p.write();
}
