#include "single_point_drivers.h"
#include "measure_drivers.h"
#include "states.h"
#include "observables.h"
#include "rng.h"

void single_point_driver(interface_2 &p)
{
	unsigned int n_observables, i;
	data_structures<double> ds;
	observables_vector_real observables;
	running_stat<double> E;
	gradient_running_stat G;
	
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

	homogeneous_state(p.dmu, p.t1, p.t2, p.t3, p.t4, p.beta, true, ds);

	G.reset(ds.n_derivatives);
	E.reset(20);
	monte_carlo_driver(p.n_measure, p.n_skip, true, observables, p.coefficients, ds, E, G);
	
	p.E << E.mean() << sqrt(E.variance_of_the_mean());
	p.G = G.gradient();
	p.sG = G.gradient_covariance();
	p.autocorrelation = E.autocorrelation();
	p.write();
}