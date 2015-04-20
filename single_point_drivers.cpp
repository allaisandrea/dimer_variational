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
	std::vector< running_stat<double> > F;
	
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

	homogeneous_state(p.dmu, p.t1, p.t2, p.t3, p.t4, p.beta, false, ds);

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