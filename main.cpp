#define __main_cpp__
#include <mpi.h>
#include <ctime>
#include "rng.h"
#include "test_minimization_gradient.cpp"
#include "tests.cpp"
#include "minimization_gradient_drivers.h"
#include "single_point_drivers.h"
#include "utilities.h"

int main(int argc, char** argv) 
{
	if(argc == 2)
	{
		time(&start_time);
		  
		interface_base p0;
		p0.read(argv[1]);
		
		if(p0.version == 1)
		{
			int rc;
  
			rc = MPI_Init(&argc, &argv);
			if(rc != MPI_SUCCESS)
				throw std::runtime_error("Failed to initialize MPI");
			
			
			interface_1 p1;
			p1.read(argv[1]);
			driver_minimize(p1);
			
			MPI_Finalize();
		}
		else if(p0.version == 2)
		{
			interface_2 p2;
			p2.read(argv[1]);
			single_point_driver(p2);
		}
		else if(p0.version == 3)
		{
			interface_3 p3;
			p3.read(argv[1]);
			qp_residual_driver(p3);
		}
	}
	else
	{
// 		test_build_graph();
// 		test_rotate_face_no_step();
// 		test_singular();
// 		test_rotate_face_with_step();
// 		test_correct_distribution();
// 		test_apriori_swap_proposal();
// 		test_swap_states();
// 		test_map();
// 		test_eigensystem_variation();
// 		test_homogeneous_state_derivatives();
// 		test_monte_carlo_driver();
// 		test_homogeneous_eigenfunctions();
// 		test_states_autocorrelation();
// 		test_rank_1_update();
// 		test_minimization();
// 		test_running_stat();
// 		test_minimization_parallel();
// 		test_distribution_1();
// 		test_gradient_stat_1();
// 		test_running_stat();
// 		test_minimization_gradient();
// 		test_build_graph();
// 		test_basis_functions();
		test_homogeneous_state(); 
	}
	return 0x0;
}

#undef __main_cpp__ 