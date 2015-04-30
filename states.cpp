#include <exception>
#include <iomanip>
#include "states.h"
#include "rng.h"
#include "linear_algebra.h"
#include "utilities.h"

const double pi = 3.1415926535897932385;

arma::mat homogeneous_state_hamiltonian(unsigned int L, const arma::vec & u)
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
		
		H(i00, i00) = -0.5 * u(0);
		H(i00 + 1, i00 + 1) = +0.5 * u(0);
		
		H(i00    , i01    ) = -u(2);
		H(i00 + 1, i10 + 1) = -u(1);
		
		H(i00    , i00 + 1) = -u(3);
		H(i00    , i10 + 1) = -u(3);
		H(i10 + 1, i01    ) = -u(3);
		H(i00 + 1, i01    ) = -u(3);
		
		H(i00, i01 + 1) = -u(5);
		H(i00, i11 + 1) = -u(5);
		H(i00 + 1, i02) = -u(5);
		H(i10 + 1, i02) = -u(5);
		H(i00 + 1, i10) = -u(4);
		H(i00 + 1, i11) = -u(4);
		H(i00, i20 + 1) = -u(4);
		H(i01, i20 + 1) = -u(4);
	}

	H += trans(H);
	return H;
}

void basis_functions(unsigned int L, unsigned int k, unsigned int q, arma::mat& psi)
{
	unsigned int x, y, L_2, i;
	double c1, c2, s1, s2, ak, bk, aq, bq, buf, norm;
	
	// c1 = cos[2 pi k / L (x + 1/2) + 2 pi q / L y]
	// c2 = cos[2 pi k / L x + 2 pi q / L (y + 1/2)]
	// s1 = sin[2 pi k / L (x + 1/2) + 2 pi q / L y]
	// s2 = sin[2 pi k / L x + 2 pi q / L (y + 1/2)]
	
	L_2 = L / 2;
	
	
	k %= L;
	c1 = cos(pi * k / L);
	s1 = sin(pi * k / L);
	ak = 2. * s1 * s1;
	bk = sin(2. * pi * k / L);
	c1 /= L;
	s1 /= L;
	
	q %= L;
	c2 = cos(pi * q / L);
	s2 = sin(pi * q / L);
	aq = 2. * s2 * s2;
	bq = sin(2. * pi * q / L);
	c2 /= L;
	s2 /= L;
	
	if(q % L_2 || k % L_2)
	{
		buf = sqrt(2.);
		c1 *= buf;
		s1 *= buf;
		c2 *= buf;
		s2 *= buf;
	}
	
	
	psi.zeros(2 * L * L, 4);
	i = 0;
	for(y = 0; y < L; y++)
	{
		for(x = 0; x < L; x++)
		{
			psi(i    , 0) = c1;
			psi(i + 1, 1) = c2;
			psi(i    , 2) = s1;
			psi(i + 1, 3) = s2;
			
			buf = c1;
			c1 -= ak * c1 + bk * s1;
			s1 -= ak * s1 - bk * buf;
			buf = c2;
			c2 -= ak * c2 + bk * s2;
			s2 -= ak * s2 - bk * buf;
			i += 2;
		}
		buf = c1;
		c1 -= aq * c1 + bq * s1;
		s1 -= aq * s1 - bq * buf;
		buf = c2;
		c2 -= aq * c2 + bq * s2;
		s2 -= aq * s2 - bq * buf;
	}
}

void kspace_hamiltonian(const arma::vec &u, unsigned int L, unsigned int ik, unsigned int iq, arma::mat& h, arma::cube& dh)
{
	double kx = 2. * pi * ik / L, ky = 2. * pi * iq / L;
	unsigned int i;
	
	dh.zeros(2, 2, 6);
	dh(0, 0, 0) = -1.;
	dh(1, 1, 0) = +1.;
	dh(1, 1, 1) = -2. * cos(kx);
	dh(0, 0, 2) = -2. * cos(ky);
	dh(0, 1, 3) = dh(1, 0, 3) = -4. * cos(0.5 * kx) * cos(0.5 * ky);
	dh(0, 1, 4) = dh(1, 0, 4) = -4. * cos(1.5 * kx) * cos(0.5 * ky);
	dh(0, 1, 5) = dh(1, 0, 5) = -4. * cos(0.5 * kx) * cos(1.5 * ky);
	
	h.zeros(2, 2);
	for(i = 0; i < 6; i++)
		h += u(i) * dh.slice(i);
}

void homogeneous_state(
	const arma::vec &x,
	const arma::mat &P,
	const arma::vec &u0,
	double beta,
	data_structures<double> &ds)
{
	unsigned int k, q, L, L_2, d, nu = 6, i, di, j;
	double s, Ds, m;
	arma::vec u, wk, vbuf;
	arma::mat h, Vk, Dwk, psik, mbuf;
	arma::cube Dh, DVk;
	
	if(P.n_cols != x.n_rows || P.n_rows != nu)
		throw std::logic_error("Transformation matrix is ill defined");
	u = P * x + u0;
	
	L = ds.L;
	L_2 = L / 2;
	
	ds.psi[0].set_size(ds.n_edges, ds.n_edges);
	ds.w[0].set_size(ds.n_edges);
	ds.Dpsi[0].set_size(ds.n_edges, ds.n_edges, x.n_rows);
	ds.Dw[0].set_size(ds.n_edges, x.n_rows);
	ds.momenta.set_size(2, ds.n_edges);
	
	DVk.set_size(2, 2, x.n_rows);
	Dwk.set_size(2, x.n_rows);
	i = 0;
	for(q = 0; q <= L_2; q++)
	for(k = 0; k < L; k++)
	{
		kspace_hamiltonian(u, L, k, q, h, Dh);
		basis_functions(L, k, q, psik);
		eig_sym(wk, Vk, h);
		
		DVk.zeros();
		Dwk.zeros();
		for(j = 0; j < nu; j++)
		{
			eigensystem_variation(Vk, wk, Dh.slice(j), mbuf, vbuf);
			for(d = 0; d < x.n_rows; d++)
			{
				DVk.slice(d) += P(j, d) * mbuf;
				Dwk.col(d) += P(j, d) * vbuf;
			}
		}
		
		if(k == 0 && q == 0)
		{
			ds.psi[0].cols(i, i + 1) = psik.cols(0, 1) * Vk;
			ds.w[0].rows(i, i + 1) = wk;
			for(d = 0; d < x.n_rows; d++)
			{
				ds.Dpsi[0].slice(d).cols(i, i + 1) = psik.cols(0, 1) * DVk.slice(d);
				ds.Dw[0].col(d).rows(i, i + 1) = Dwk.col(d);
			}
			di = 2;
		}
		else if(k == L_2 && q == L_2)
		{
			ds.psi[0].cols(i, i + 1) = psik.cols(2, 3) * Vk;
			ds.w[0].rows(i, i + 1) = wk;
			for(d = 0; d < x.n_rows; d++)
			{
				ds.Dpsi[0].slice(d).cols(i, i + 1) = psik.cols(2, 3) * DVk.slice(d);
				ds.Dw[0].col(d).rows(i, i + 1) = Dwk.col(d);
			}
			di = 2;
		}
		else if(k == L_2 && q == 0)
		{
			ds.psi[0].col(i) = psik.col(1);
			ds.psi[0].col(i + 1) = psik.col(2);
			ds.w[0](i) = wk(1);
			ds.w[0](i + 1) = wk(0);
			for(d = 0; d < x.n_rows; d++)
			{
				ds.Dpsi[0].slice(d).cols(i, i + 1).zeros();
				ds.Dw[0](i,     d) = Dwk(1, d);
				ds.Dw[0](i + 1, d) = Dwk(0, d);
			}
			di = 2;
		}
		else if(k == 0 && q == L_2)
		{
			ds.psi[0].col(i) = psik.col(0);
			ds.psi[0].col(i + 1) = psik.col(3);
			ds.w[0](i) = wk(0);
			ds.w[0](i + 1) = wk(1);
			for(d = 0; d < x.n_rows; d++)
			{
				ds.Dpsi[0].slice(d).cols(i, i + 1).zeros();
				ds.Dw[0](i,     d) = Dwk(0, d);
				ds.Dw[0](i + 1, d) = Dwk(1, d);
			}
			di = 2;
		}
		else
		{
			ds.psi[0].cols(i + 0, i + 1) = psik.cols(0, 1) * Vk;
			ds.psi[0].cols(i + 2, i + 3) = psik.cols(2, 3) * Vk;
			ds.w[0].rows(i + 0, i + 1) = wk;
			ds.w[0].rows(i + 2, i + 3) = wk;
			for(d = 0; d < x.n_rows; d++)
			{
				ds.Dpsi[0].slice(d).cols(i + 0, i + 1) = psik.cols(0, 1) * DVk.slice(d);
				ds.Dpsi[0].slice(d).cols(i + 2, i + 3) = psik.cols(2, 3) * DVk.slice(d);
				ds.Dw[0].col(d).rows(i + 0, i + 1) = Dwk.col(d);
				ds.Dw[0].col(d).rows(i + 2, i + 3) = Dwk.col(d);
			}
			di = 4;
		}
		
		for(j = 0; j < di; j++)
		{
			ds.momenta(0, i + j) = k;
			ds.momenta(1, i + j) = q;
		}
		
		i += di;
		if(q % L_2 == 0 && k == L_2)
			break;
	}
	
	s = stddev(ds.w[0]);
	m = mean(ds.w[0]);
	for(d = 0; d < x.n_rows; d++)
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
	ds.Dphi.zeros(2 * L * L, x.n_rows);
	ds.n_derivatives = x.n_rows;
}
