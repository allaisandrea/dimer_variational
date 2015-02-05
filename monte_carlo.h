#ifndef __monte_carlo_h__
#define __monte_carlo_h__

#include "data_structures.h"

template<class type>
void initial_configuration(unsigned int Nf, data_structures<type> &ds);

template <class type>
unsigned int rotate_face(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	data_structures<type> &ds);
#endif