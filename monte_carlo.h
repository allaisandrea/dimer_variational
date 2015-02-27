#ifndef __monte_carlo_h__
#define __monte_carlo_h__

#include "data_structures.h"

template<class type>
void initial_configuration(unsigned int Nu, unsigned int Nd, data_structures<type> &ds);

template <class type>
unsigned int rotate_face(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	double &amp,
	data_structures<type> &ds);

bool apriori_swap_proposal(const arma::vec& w, const arma::uvec &Jo, const arma::uvec &Je, unsigned int &io, unsigned int &ie);
template <class type>
bool swap_states(unsigned int s, double& amp, data_structures<type> & ds);


#endif