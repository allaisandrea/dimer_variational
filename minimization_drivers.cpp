#include <stdexcept>
#include <ctime>
#include <mpi.h>
#include "minimization_drivers.h"
#include "measure_drivers.h"
#include "minimization.h"
#include "observables.h"
#include "rng.h"
#include "states.h"
#include "utilities.h"


void min_f_internal(const arma::vec& x, running_stat<double> & y, interface_1 &p)
{
	unsigned int n_observables;
	data_structures<double> ds;
	observables_vector_real observables;
	
	ds.L = p.L;
	ds.Nf[0] = p.Nu;
	ds.Nf[1] = p.Nd;  
	
	build_graph(ds);
	
// 	observables.push_back(&boson_hopping);
// 	observables.push_back(&boson_potential);
	observables.push_back(&fermion_hopping_1);
	observables.push_back(&fermion_hopping_2);
	observables.push_back(&fermion_hopping_3);
	n_observables = observables.size();
	
	homogeneous_state(0., 1., x(0), x(1), x(2), p.beta, false, ds);
	
	monte_carlo_driver(p.n_measure, p.n_skip, observables, p.coefficients, ds, y);
}

void min_f(const arma::vec& _x, running_stat<double> & y, interface_1 &p)
{
	int i, n_threads;
	unsigned int seed_i;
	arma::vec x(_x);
	running_stat<double> y1;  
	
	std::cout << elapsed_time_string() << " \tFunction evaluation" << std::endl;
	std::cout << elapsed_time_string() << " \t\tx: " << trans(x);
	
	
	MPI_Comm_size(MPI_COMM_WORLD,&n_threads);
	
	for(i = 1; i < n_threads; i++)
		MPI_Send(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, i, 1, MPI_COMM_WORLD);
	
	min_f_internal(x, y, p);
	
	for(i = 1; i < n_threads; i++)
	{
		MPI_Recv(&y1, sizeof(y1), MPI_BYTE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		y(y1);
	}
	
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
		arma::mat x0;
		
		std::cout << asctime(localtime(&start_time)) << std::endl;
		std::cout << "Using " << n_threads << " CPUs\n" << std::endl;
		
		rng::seed(p.seed);
		
		for(i = 1; i < n_threads; i++)
		{
			seed = rng::get(); 
			MPI_Send(&seed, sizeof(seed), MPI_BYTE, i, 1, MPI_COMM_WORLD);
		}
		x0.zeros(n_dim, n_dim + 1);
		x0.col(0) = p.x0;
		for(i = 0; i < n_dim; i++)
		{
			x0.col(i + 1) = p.x0;
			x0(i, i + 1) += p.dx;
		}
		simplex_minimize<interface_1>(x0, min_f, p);
	}
	else
	{
		arma::vec x;
		running_stat<double> y;
		
		MPI_Recv(&seed, sizeof(p.seed), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		rng::seed(seed);
		
		x.set_size(n_dim);
		while(true)
		{
			MPI_Recv(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			y.reset();
			min_f_internal(x, y, p);
			
			MPI_Send(&y, sizeof(y), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
		}
		
	}
}