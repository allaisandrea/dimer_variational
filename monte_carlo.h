#ifndef __monte_carlo_h__
#define __monte_carlo_h__

#include "data_structures.h"

template<class type>
void initial_configuration(data_structures<type> &ds);

template <class type>
unsigned int monte_carlo_step(
	bool step,
	bool swap_states,
	double &amp,
	data_structures<type> &ds);
#endif