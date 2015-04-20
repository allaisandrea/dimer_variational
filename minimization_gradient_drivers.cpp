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


void min_f_internal(const arma::vec& x, running_stat<double> & y, interface_1 &p)
{
	unsigned int i, n_observables;
	data_structures<double> ds;
	observables_vector_real observables;
	gradient_running_stat dummy;
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
	
	homogeneous_state(0., x(0), 1., x(1), x(2), p.beta, false, ds);
	
	monte_carlo_driver(p.n_measure, p.n_skip, false, observables, p.coefficients, ds, y, dummy);
}

void min_df_internal(const arma::vec& x, running_stat<double> & y, gradient_running_stat &G, interface_1 &p)
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
	
	homogeneous_state(0., x(0), 1., x(1), x(2), p.beta, true, ds);
	
	monte_carlo_driver(p.n_measure, p.n_skip, true, observables, p.coefficients, ds, y, G);
}

void min_f(const arma::vec& _x, running_stat<double> & y, interface_1 &p)
{
	int i, n_threads, ido;
	arma::vec x(_x);
	running_stat<double> y1;  
	
	std::cout << elapsed_time_string() << " \tFunction evaluation" << std::endl;
	std::cout << elapsed_time_string() << " \t\tx: " << trans(x);
	
	
	MPI_Comm_size(MPI_COMM_WORLD,&n_threads);
	
	ido = 0;
	for(i = 1; i < n_threads; i++)
	{
		MPI_Send(&ido, sizeof(ido), MPI_BYTE, i, 1, MPI_COMM_WORLD);
		MPI_Send(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, i, 1, MPI_COMM_WORLD);
	}
	
	min_f_internal(x, y, p);
	
	for(i = 1; i < n_threads; i++)
	{
		MPI_Recv(&y1, sizeof(y1), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		y(y1);
	}
	
	std::cout << elapsed_time_string() << " \t\tf: " << y.mean() << " +- " << sqrt(y.variance_of_the_mean()) << " @ " << y.count() << std::endl;
}

void min_df(const arma::vec& _x, running_stat<double> & y, arma::vec &G, interface_1 &p)
{
	int i, n_threads, ido;
	arma::vec x(_x), Gi;
	running_stat<double> y1;  
	gradient_running_stat Gstat;
	
	std::cout << elapsed_time_string() << " \tFunction evaluation" << std::endl;
	std::cout << elapsed_time_string() << " \t\tx: " << trans(x);
	
	
	MPI_Comm_size(MPI_COMM_WORLD,&n_threads);
	
	ido = 1;
	for(i = 1; i < n_threads; i++)
	{
		MPI_Send(&ido, sizeof(ido), MPI_BYTE, i, 1, MPI_COMM_WORLD);
		MPI_Send(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, i, 1, MPI_COMM_WORLD);
	}	
	Gstat.reset(5);
	min_df_internal(x, y, Gstat, p);
	G = Gstat.gradient();
	
	Gi.set_size(G.n_elem);
	for(i = 1; i < n_threads; i++)
	{
		MPI_Recv(&y1, sizeof(y1), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		y(y1);
		MPI_Recv(Gi.memptr(), Gi.n_elem * sizeof(double), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		G += (Gi - G) / (i + 1);
	}
	
	G << G(1) << G(3) << G(4);
	
	std::cout << elapsed_time_string() << " \t\tf: " << y.mean() << " +- " << sqrt(y.variance_of_the_mean()) << " @ " << y.count() << std::endl;
}

void driver_minimize(interface_1 &p)
{
	int rc, rank, i, n_dim, n_threads;
	unsigned int seed;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD,&n_threads);
	
	n_dim = 3;
	if(rank == 0)
	{
		running_stat <double> v;
		std::cout << asctime(localtime(&start_time)) << std::endl;
		std::cout << "Using " << n_threads << " CPUs\n" << std::endl;
		
		rng::seed(p.seed);
		
		for(i = 1; i < n_threads; i++)
		{
			seed = rng::get(); 
			MPI_Send(&seed, sizeof(seed), MPI_BYTE, i, 1, MPI_COMM_WORLD);
		}
		conjugate_gradient<interface_1>::minimize(p.x0, v, p.step, p.accuracy, min_f, min_df, p);
	}
	else
	{
		unsigned int ido;
		arma::vec x;
		running_stat<double> y;
		gradient_running_stat Gstat;
		arma::vec G;
		
		MPI_Recv(&seed, sizeof(p.seed), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		rng::seed(seed);
		
		x.set_size(n_dim);
		while(true)
		{
			MPI_Recv(&ido, sizeof(ido), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			y.reset();
			Gstat.reset(5);
			if(ido == 0)
			{
				min_f_internal(x, y, p);
				
				MPI_Send(&y, sizeof(y), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
			}
			else if(ido == 1)
			{
				min_df_internal(x, y, Gstat, p);
				G = Gstat.gradient();
				MPI_Send(&y, sizeof(y), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
				MPI_Send(G.memptr(), G.n_elem * sizeof(double), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
			}
			
		}
		
	}
}