#ifndef __monte_carlo_h__
#define __monte_carlo_h__

#include "data_structures.h"

template<class type>
void compute_state_weights(data_structures<type> &ds);

template<class type>
void initial_configuration(data_structures<type> &ds);

template <class type>
unsigned int rotate_face(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	double &amp,
	data_structures<type> &ds);

template<class type>
bool apriori_swap_proposal(unsigned int s, const data_structures<type> &ds, unsigned int &io, double &Zo2, unsigned int &ie, double &Ze2);

template <class type>
bool swap_states(unsigned int s, double& amp, data_structures<type> & ds);


#endif