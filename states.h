#ifndef __states_h__
#define __states_h__

#include "data_structures.h"

void homogeneous_state(
	const arma::vec &x,
	const arma::mat &P,
	const arma::vec &u0,
	double beta,
	data_structures<double> &ds);

#endif
