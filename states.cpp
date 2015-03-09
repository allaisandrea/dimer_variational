#include <exception>
#include "states.h"
#include "rng.h"
#include "linear_algebra.h"
#include "utilities.h"

arma::mat homogeneous_state_hamiltonian(
	unsigned int L,
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4)
{
	unsigned int x, y, i00, i10, i11, i01, i20, i02;
	arma::mat H;
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
	return H;
}

inline void step_trig(double &c, double &s, double a, double b)
{
	static double buf;
	buf = c;
	c -= a * c + b * s;
	s -= a * s - b * buf;
}

arma::mat plane_waves(unsigned int L)
{
	const double pi = 4. * atan(1.);
	unsigned int x, y, k, q, i, j;
	double ak, bk, aq, bq, cx, sx, cx1, sx1, cy, sy, cy1, sy1, nm;
	arma::mat U;
	
	if(L % 2 != 0)
		throw std::logic_error("Side must be even");
	
	U.zeros(2 * L * L, 2 * L * L + 8 * L + 8);
	
	j = 0;
	for(q = 0; q <= L / 2; q++)
	for(k = 0; k <= L / 2; k++)
	{
		ak = sin(pi * k / L);
		ak = 2. * ak * ak;
		bk = sin(2. * pi * k / L);
		
		aq = sin(pi * q / L);
		aq = 2. * aq * aq;
		bq = sin(2. * pi * q / L);
		
		cx = 1.; 
		sx = 0.;
		
		cx1 = cos(pi * k / L);
		sx1 = sin(pi * k / L);
		
		cy = 1.; 
		sy = 0.;
		
		cy1 = cos(pi * q / L);
		sy1 = sin(pi * q / L);
		
		i = 0;
		for(y = 0; y < L; y++)
		{
			for(x = 0; x < L; x++)
			{
				U(i    , j    ) = cx1 * cy;
				U(i + 1, j + 1) = cx  * cy1;
				U(i    , j + 2) = sx1 * cy;
				U(i + 1, j + 3) = sx  * cy1;
				U(i    , j + 4) = cx1 * sy;
				U(i + 1, j + 5) = cx  * sy1;
				U(i    , j + 6) = sx1 * sy;
				U(i + 1, j + 7) = sx  * sy1;
				
				step_trig(cx,  sx,  ak, bk);
				step_trig(cx1, sx1, ak, bk);
				i+=2;
			}
			step_trig(cy,  sy,  aq, bq);
			step_trig(cy1, sy1, aq, bq);
		}
		
		j += 8;
	}
	
	for(j = 0; j < U.n_cols; j++)
	{
		nm = norm(U.col(j), "fro");
		if(nm > 1.e-7)
			U.col(j) /= nm;
	}
	return U;
}

template<class type>
void homogeneous_state(
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4,
	double beta,
	data_structures<type> &ds)
{
	const double pi = 3.141592653589793;
	unsigned int i, j, c, k, q, L, d;
	double kk, qq, nm1, nm2, s, Ds, m;
	arma::cube h, v;
	arma::mat U, H, wk;
	arma::vec vbuf;
	
	L = ds.L;
	
	U = plane_waves(L);
	ds.w[0].set_size(2 * L * L);
	ds.psi[0].set_size(2 * L * L, 2 * L * L);
	ds.Dw[0].set_size(2 * L * L, 5);
	ds.Dpsi[0].set_size(2 * L * L, 2 * L * L, 5);
	
	
	h.set_size(2, 2, 6);
	v.set_size(2, 2, 6);
	wk.set_size(2, 6);
	
	j = 0;
	c = 0;
	
	for(q = 0; q <= L/2; q++)
	for(k = 0; k <= L/2; k++)
	{
		kk = 2. * pi * k / L;
		qq = 2. * pi * q / L;
		
		h(0, 0, 1) = + 0.5;
		h(1, 1, 1) = - 0.5;
		h(1, 0, 1) = 0.;
		h(0, 1, 1) = 0.;
		
		h(0, 0, 2) = -2 * cos(qq);
		h(1, 1, 2) = -2 * cos(kk);
		h(1, 0, 2) = 0.;
		h(0, 1, 2) = 0.;
		
		h(0, 0, 3) = 0.;
		h(1, 1, 3) = 0.;
		h(1, 0, 3) = -4 * cos(0.5 * kk) * cos(0.5 * qq);
		h(0, 1, 3) = h(1, 0, 3);
		
		h(0, 0, 4) = 0.;
		h(1, 1, 4) = 0.;
		h(1, 0, 4) = - 4. * (cos(0.5 * kk) * cos(1.5 * qq) + cos(0.5 * qq) * cos(1.5 * kk));
		h(0, 1, 4) = h(1, 0, 4);
		
		h(0, 0, 5) = -2 * cos(kk);
		h(1, 1, 5) = -2 * cos(qq);
		h(1, 0, 5) = 0.;
		h(0, 1, 5) = 0.;
		
		h.slice(0) = dmu * h.slice(1) + t1 * h.slice(2) + t2 * h.slice(3) + t3 * h.slice(4) + t4 * h.slice(5);
		
		eig_sym(vbuf, v.slice(0), h.slice(0));
		
		wk.col(0) = vbuf;
		for(d = 1; d < 6; d++)
		{
			eigensystem_variation(v.slice(0), wk.col(0), h.slice(d), v.slice(d), vbuf);
			v.slice(d) = trans(v.slice(0)) * v.slice(d);
			wk.col(d) = vbuf;
		}
		
		for(i = 0; i < 4; i++)
		{
			vbuf = U.col(j);
			U.col(j    ) = v(0, 0, 0) * U.col(j) + v(1, 0, 0) * U.col(j + 1);
			U.col(j + 1) = v(0, 1, 0) * vbuf     + v(1, 1, 0) * U.col(j + 1);
			
			if(norm(U.col(j), "fro") > 1.e-7)
			{
				ds.w[0](c) = wk(0, 0);
				ds.psi[0].col(c) = arma::conv_to<arma::Col<type> >::from(U.col(j));
				for(d = 0; d < 5; d++)
				{
					ds.Dw[0](c, d) = wk(0, d + 1);
					vbuf = v(0, 0, d + 1) * U.col(j) + v(1, 0, d + 1) * U.col(j + 1);
					ds.Dpsi[0].slice(d).col(c) = arma::conv_to<arma::Col<type> >::from(vbuf);
				}
				c++;
			}
			if(norm(U.col(j + 1), "fro") > 1.e-7)
			{
				ds.w[0](c) = wk(1, 0);
				ds.psi[0].col(c) = arma::conv_to<arma::Col<type> >::from(U.col(j + 1));
				for(d = 0; d < 5; d++)
				{
					ds.Dw[0](c, d) = wk(1, d + 1);
					vbuf = v(0, 1, d + 1) * U.col(j) + v(1, 1, d + 1) * U.col(j + 1);
					ds.Dpsi[0].slice(d).col(c) = arma::conv_to<arma::Col<type> >::from(vbuf);
				}
				c++; 
			}
			j+=2;
		}
		
	}
	
	s = stddev(ds.w[0]);
	m = mean(ds.w[0]);
	for(d = 0; d < 5; d++)
	{
		Ds = accu((ds.w[0] - m) % ds.Dw[0].col(d)) / (ds.w[0].n_rows - 1);
		ds.Dw[0].col(d) = beta * (ds.Dw[0].col(d) / s - ds.w[0] * Ds / s / s / s);
	}
	ds.w[0] *= beta / s;
	
	ds.psi[1] = ds.psi[0];
	ds.w[1] = ds.w[0];
	ds.Dpsi[1] = ds.Dpsi[0];
	ds.Dw[1] = ds.Dw[0];
	ds.phi.ones(2 * L * L);
	ds.Dphi.zeros(2 * L * L, 5);
	ds.n_derivatives = 5;
	
}


template
void homogeneous_state<double>(
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4, 
	double beta,
	data_structures<double> &ds);

template
void homogeneous_state<arma::cx_double>(
	double dmu, 
	double t1, 
	double t2, 
	double t3,
	double t4, 
	double beta,
	data_structures<arma::cx_double> &ds);
