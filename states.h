#ifndef __states_h__
#define __states_h__

#include "data_structures.h"
template<class type>
void homogeneous_state(
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4, 
	double beta,
	data_structures<type> &ds);

arma::mat homogeneous_state_hamiltonian(
	unsigned int L,
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4);
arma::mat plane_waves(unsigned int L);
#endif
