#define __main_cpp__
#include <mpi.h>
#include "rng.h"
#include "tests.cpp"


int main(int argc, char** argv)
{
	int rc;
	rc = MPI_Init(&argc, &argv);
	if(rc != MPI_SUCCESS)
		throw std::runtime_error("Failed to initialize MPI");
// 	test_build_graph();
// 	test_homogeneous_state(); 
// 	test_rotate_face_no_step();
// 	test_singular();
// 	test_rotate_face_with_step();
// 	test_correct_distribution();
// 	test_apriori_swap_proposal();
// 	test_swap_states();
// 	test_map();
// 	test_eigensystem_variation();
// 	test_homogeneous_state_derivatives();
// 	test_monte_carlo_driver();
// 	test_homogeneous_eigenfunctions();
// 	test_states_autocorrelation();
// 	test_rank_1_update();
// 	test_minimization();
// 	test_running_stat();
	test_minimization_parallel();
	
	MPI_Finalize();
	return 0x0;
}

#undef __main_cpp__ 