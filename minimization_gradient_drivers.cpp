#include <stdexcept>
#include <ctime>
#include <mpi.h>
#include "minimization_gradient_drivers.h"
#include "measure_drivers.h"
#include "minimization_gradient.h"
#include "observables.h"
#include "rng.h"
#include "states.h"
#include "utilities.h"


void min_df_internal(const arma::vec& x, running_stat & y, gradient_running_stat &g, interface_1 &p)
{
	unsigned int i, n_observables;
	data_structures<double> ds;
	observables_vector_real observables;
	ds.L = p.L;
	ds.Nf[0] = p.Nu;
	ds.Nf[1] = p.Nd;  
	
	build_graph(ds);
	
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
	
	homogeneous_state(0., x(0), 1., x(1), x(2), p.beta, 0x1A, ds);
	
	monte_carlo_driver(p.n_measure, p.n_skip, true, observables, p.coefficients, ds, y, g);
}


void min_df(const arma::vec& _x, running_stat & y, gradient_running_stat &g, interface_1 &p)
{
	unsigned int stop;
	int i, n_threads;
	arma::vec x(_x);
	running_stat y1;  
	gradient_running_stat g1;
	
// 	std::cout << elapsed_time_string() << " \tFunction evaluation" << std::endl;
// 	std::cout << elapsed_time_string() << " \t\tx: " << trans(x);
	
	MPI_Comm_size(MPI_COMM_WORLD,&n_threads);
	
	for(i = 1; i < n_threads; i++)
	{
		stop = 0;
		MPI_Send(&stop, 1, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD);
		MPI_Send(x, i, 1, MPI_COMM_WORLD);
	}
	
	min_df_internal(x, y, g, p);
	
	for(i = 1; i < n_threads; i++)
	{
		MPI_Recv(y1, i, 1, MPI_COMM_WORLD);
		MPI_Recv(g1, i, 1, MPI_COMM_WORLD);
		y(y1);
		g(g1);
	}
	
// 	std::cout << elapsed_time_string() << " \t\tf: " << y.mean() << " +- " << sqrt(y.variance_of_the_mean()) << " @ " << y.count() << std::endl;
}

void driver_minimize(interface_1 &p)
{
	int rc, rank, i, n_dim, n_threads;
	unsigned int seed, stop;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&n_threads);
	
	if(rank == 0)
	{
		running_stat  v;
		std::cout << asctime(localtime(&start_time)) << std::endl;
		std::cout << "Using " << n_threads << " CPUs\n" << std::endl;
		
		rng::seed(p.seed);
		
		for(i = 1; i < n_threads; i++)
		{
			seed = rng::get(); 
			MPI_Send(&seed, 1, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD);
		}
		
		conjugate_gradient<interface_1>::minimize(p.x0, v, p.step, p.max_count, min_df, p);
		
		for(i = 1; i < n_threads; i++)
		{
			stop = 1;
			MPI_Send(&stop, 1, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD);
		}
	}
	else
	{
		arma::vec x;
		running_stat y;
		gradient_running_stat g;
		
		MPI_Recv(&seed, 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		rng::seed(seed);
		
		stop = 0;
		while(!stop)
		{
			MPI_Recv(&stop, 1, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(!stop)
			{
				MPI_Recv(x, 0, 1, MPI_COMM_WORLD);
				
				y.reset();
				g.reset(x.n_rows);
				min_df_internal(x, y, g, p);
				
				MPI_Send(y, 0, 1, MPI_COMM_WORLD);
				MPI_Send(g, 0, 1, MPI_COMM_WORLD);
			}
		}
	}
}