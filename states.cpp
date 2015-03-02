#include <exception>
#include "states.h"
#include "rng.h"
template<class type>
void homogeneous_state(
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4,
	data_structures<type> &ds)
{
	unsigned int x, y, L, i00, i01, i10, i11, i20, i02, i, Pi;
	arma::mat H, Px, Py, psi;
	L = ds.L;
	H.zeros(2 * L * L, 2 * L * L);
	for(x = 0; x < L; x++)
	for(y = 0; y < L; y++)
	{
		i00 = 2 * (x + L * y);
		i10 = 2 * ((x + 1) % L + L * y);
		i01 = 2 * (x + L * ((y + 1) % L));
		i11 = 2 * ((x + 1) % L + L * ((y + 1) % L));
		i20 = 2 * ((x + 2) % L + L * y);
		i02 = 2 * (x + L * ((y + 2) % L));
		
		H(i00, i00) = 0.25 * dmu;
		H(i00 + 1, i00 + 1) = -0.25 * dmu;
		
		H(i00    , i01    ) = -t1;
		H(i00 + 1, i10 + 1) = -t1;
		
		H(i00    , i00 + 1) = -t2;
		H(i00    , i10 + 1) = -t2;
		H(i10 + 1, i01    ) = -t2;
		H(i00 + 1, i01    ) = -t2;
		
		H(i00, i01 + 1) = -t3;
		H(i00, i11 + 1) = -t3;
		H(i00 + 1, i02) = -t3;
		H(i10 + 1, i02) = -t3;
		H(i00 + 1, i10) = -t3;
		H(i00 + 1, i11) = -t3;
		H(i00, i20 + 1) = -t3;
		H(i01, i20 + 1) = -t3;
		
		H(i00, i10) = - t4;
		H(i00 + 1, i01 + 1) = - t4;
	}

	H += trans(H);
	
	Px.zeros(2 * L * L, 2 * L * L);
	Py.zeros(2 * L * L, 2 * L * L);
	for(x = 0; x < L; x++)
	for(y = 0; y < L; y++)
	{
		i = 2 * (x + L * y);
		
		Pi = 2 * ((L - x - 1) % L + L * y);
		Px(i, Pi) = Px(Pi, i) = 1;
		Pi = 2 * (x + L * ((L - y) % L));
		Py(i, Pi) = Py(Pi, i) = 1;
		
		i = 2 * (x + L * y) + 1;
		
		Pi = 2 * ((L - x) % L + L * y) + 1;
		Px(i, Pi) = Px(Pi, i) = 1;
		Pi = 2 * (x + L * ((L - y - 1) % L)) + 1;
		Py(i, Pi) = Py(Pi, i) = 1;
	}
	
	arma::eig_sym(ds.w[0], psi, H + rng::gaussian() * Px + rng::gaussian() * Py);
	
	for(i = 0; i + 1 < ds.w[0].n_elem; i++)
		if(fabs(ds.w[0](i) - ds.w[0](i + 1)) < 1.e-8)
			throw std::runtime_error("Degeneracy has not been fully lifted");
	
	for(i = 0; i < 2 * L * L; i++)
		ds.w[0](i) = arma::dot(psi.col(i), H * psi.col(i));
	ds.w[1] = ds.w[0];
	
	ds.psi[1] = ds.psi[0] = arma::conv_to<arma::Mat<type> >::from(psi);
	
	ds.phi.ones(2 * L * L);
}


template
void homogeneous_state<double>(
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4, 
	data_structures<double> &ds);

template
void homogeneous_state<arma::cx_double>(
	double dmu, 
	double t1, 
	double t2, 
	double t3,
	double t4, 
	data_structures<arma::cx_double> &ds);
