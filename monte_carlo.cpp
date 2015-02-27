#include <iomanip>
#include "monte_carlo.h"
#include "rng.h"
#include "linear_algebra.h"
#include "utilities.h"

// #define lock_face_bb
// #define lock_face_bf
// #define lock_face_ff

template<class type>
void initial_configuration(unsigned int Nu, unsigned int Nd, data_structures<type> &ds)
{
	unsigned int x, y, L, c, s, max_attempts;
	double dummy;
	arma::uvec edges, e;
	
	ds.Nf[0] = Nu;
	ds.Nf[1] = Nd;
	L = ds.L;
	
	if(L % 2 == 1)
		throw "Lattice side must be even";
	
	edges.set_size(L * L / 2);
	
	c = 0;
	for(y = 0; y < L; y++)
	for(x = 0; x < L; x+=2)
	{
		edges(c) = 2 * (x + L * y);
		c++;
	}
	edges = shuffle(edges);
	
	if(edges.n_elem < Nu + Nd)
		throw "Too many fermions";
	
	ds.particles.zeros(2 * L * L);
	ds.M[0].set_size(Nu, Nu);
	ds.M[1].set_size(Nd, Nd);
	
	for(s = 0; s < 2; s++)
	{
		ds.J[s] = sort_index(ds.w[s]);
		ds.K[s] = ds.J[s].rows(ds.Nf[s], ds.J[s].n_rows - 1);
		ds.J[s].resize(ds.Nf[s]);
	}
	
	for(c = 0; c < Nu + Nd; c++)
	{
		ds.particles(edges(c)) = c + 2;
		e << edges(c);
		c < Nu ? s = 0 : s = 1;
		ds.M[s].row(c - s * Nu) = ds.psi[s](e, ds.J[s]);
	}
	
	for(c = Nu + Nd; c < edges.n_elem; c++)
	{
		ds.particles(edges(c)) = 1;
	}
	
	c = 0;
	max_attempts = 10;
	while(c < max_attempts && (singular(ds.M[0]) || singular(ds.M[1])))
	{
		while(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), false, dummy, ds) < 2);
		c++;
	}
	if(c == max_attempts)
		throw "Unable to find regular configuration";
	std::cout << "found regular configuration after " << c << " rotations.\n";
	
	ds.Mi[0] = arma::inv(ds.M[0]);
	ds.Mi[1] = arma::inv(ds.M[1]);
}

template <class type>
unsigned int rotate_face(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	double &amp,
	data_structures<type> &ds)
{
	arma::uvec particles, edge;
	edge = ds.face_edges.col(f);
	particles = ds.particles(edge);
	if(particles(0) == 0 && particles(2) == 0)
	{
		if     (particles(1) == 1 && particles(3) == 1)
			return rotate_face_bb(edge(1), edge(2), edge(3), edge(0), step, amp, ds);
		else if(particles(1) == 1 && particles(3) >= 2)
			if(clockwise)
				return rotate_face_bf(edge(1), edge(0), edge(3), edge(2), step, amp, ds);
			else
				return rotate_face_bf(edge(1), edge(2), edge(3), edge(0), step, amp, ds);
		else if(particles(1) >= 2 && particles(3) == 1)
			if(clockwise)
				return rotate_face_bf(edge(3), edge(2), edge(1), edge(0), step, amp, ds);
			else
				return rotate_face_bf(edge(3), edge(0), edge(1), edge(2), step, amp, ds);
		else if(particles(1) >= 2 && particles(3) >= 2)
			if(clockwise)
				return rotate_face_ff(edge(1), edge(0), edge(3), edge(2), step, amp, ds);
			else
				return rotate_face_ff(edge(1), edge(2), edge(3), edge(0), step, amp, ds);
	}
	else if(particles(1) == 0 && particles(3) == 0)
	{
		if     (particles(0) == 1 && particles(2) == 1)
			return rotate_face_bb(edge(0), edge(1), edge(2), edge(3), step, amp, ds);
		else if(particles(0) == 1 && particles(2) >= 2)
			if(clockwise)
				return rotate_face_bf(edge(0), edge(3), edge(2), edge(1), step, amp, ds);
			else
				return rotate_face_bf(edge(0), edge(1), edge(2), edge(3), step, amp, ds);
		else if(particles(0) >= 2 && particles(2) == 1)
			if(clockwise)
				return rotate_face_bf(edge(2), edge(1), edge(0), edge(3), step, amp, ds);
			else
				return rotate_face_bf(edge(2), edge(3), edge(0), edge(1), step, amp, ds);
		else if(particles(0) >= 2 && particles(2) >= 2)
			if(clockwise)
				return rotate_face_ff(edge(0), edge(3), edge(2), edge(1), step, amp, ds);
			else
				return rotate_face_ff(edge(0), edge(1), edge(2), edge(3), step, amp, ds);
	}
	
	return 0;
}


template<class type>
unsigned int rotate_face_bb(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	double &amp,
	data_structures<type> & ds)
{
#ifdef lock_face_bb
	return 0;
#endif
	bool accept = false;
	
	if(ds.particles(origin1) != 1 || ds.particles(origin2) != 1)
		throw "logic error: this should never happen";
	
	
	if(step)
	{
		amp = abs_squared(ds.phi(destination1) * ds.phi(destination2) / ds.phi(origin1) / ds.phi(origin2));
		if(amp > rng::uniform())
		{
// 			std::cout << std::setw(12) << amp;
			accept = true;
		}
	}
	if(!step || accept)
	{
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
		return 1;
	}
	return 0;
	
}

template<class type>
unsigned int rotate_face_bf(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	double & amp,
	data_structures<type> & ds)
{
#ifdef lock_face_bf
	return 0;
#endif
	bool accept = false;
	unsigned int p, s;

	type det;
	arma::Col<type> U;
	arma::Row<type> V;
	
	p = ds.particles(origin2) - 2;
	
	if(ds.particles(origin1) != 1 || p > ds.Nf[0] + ds.Nf[1])
		throw "logic error: this should never happen";
	
	if(p < ds.Nf[0])
		s = 0;
	else
	{
		p -=  ds.Nf[0];
		s = 1;
	}
	
	V = ds.psi[s](destination2 * arma::ones<arma::uvec>(1), ds.J[s]);
	V -= ds.psi[s](origin2 * arma::ones<arma::uvec>(1), ds.J[s]);
	
	if(step)
	{
		
		U.zeros(ds.Nf[s]);
		U(p, 0) = 1;
		U = ds.Mi[s] * U;
		
		det = 1. + dot(V, U);
		amp = abs_squared(det * ds.phi(destination1) / ds.phi(origin1));
		if(amp > rng::uniform())
		{
			accept = true;
			ds.Mi[s] -= (U * V * ds.Mi[s]) / det;
		}

	}
	if(!step || accept)
	{
		ds.M[s].row(p) += V;
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
		return 2;
	}
	return 0;
	
}


template<class type>
unsigned int rotate_face_ff(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	double &amp,
	data_structures<type> & ds)
{
#ifdef lock_face_ff
	return 0;
#endif
	bool accept = false;
	unsigned int p1, p2,  s1, s2, return_value;
	type det[2];
	arma::Mat<type> U[2], V[2], K;
	arma::uvec e;
	
	p1 = ds.particles(origin1) - 2;
	p2 = ds.particles(origin2) - 2;
	
	if(p1 > ds.Nf[0] + ds.Nf[1])
		throw "logic error: this should never happen";
	if(p2 > ds.Nf[0] + ds.Nf[1])
		throw "logic error: this should never happen";
	
	if(p1 < ds.Nf[0])
		s1 = 0;
	else
	{
		s1 = 1;
		p1 -= ds.Nf[0];
	}
	
	if(p2 < ds.Nf[0])
		s2 = 0;
	else
	{
		s2 = 1;
		p2 -= ds.Nf[0];
	}
	

	if(s1 == s2)
	{
		
		e << destination1 << destination2;
		V[s1] = ds.psi[s1](e, ds.J[s1]);
		e << origin1 << origin2;
		V[s1] -= ds.psi[s1](e, ds.J[s1]);
			
		if(step)
		{

			U[s1].zeros(ds.Nf[s1], 2);
			U[s1](p1, 0) = 1;
			U[s1](p2, 1) = 1;
			U[s1] = ds.Mi[s1] * U[s1];
			K = V[s1] * U[s1];
			K(0, 0) += 1;
			K(1, 1) += 1;
			det[s1] = K(0, 0) * K(1, 1) - K(0, 1) * K(1, 0);
			amp = abs_squared(det[s1]);
			if(amp > rng::uniform())
			{
				accept = true;
				ds.Mi[s1] -= U[s1] * inv(K) * V[s1] * ds.Mi[s1];
			}
		}
		if(!step || accept)
		{
			return_value = 3;
			ds.M[s1].row(p1) += V[s1].row(0);
			ds.M[s2].row(p2) += V[s1].row(1);
		}
	}
	else
	{
		
		e << destination1;
		V[s1] =  ds.psi[s1](e, ds.J[s1]);
		e << origin1;
		V[s1] -= ds.psi[s1](e, ds.J[s1]);
		
		e << destination2;
		V[s2] =  ds.psi[s2](e, ds.J[s2]);
		e << origin2;
		V[s2] -= ds.psi[s2](e, ds.J[s2]);
		
		if(step)
		{
			U[s1].zeros(ds.Nf[s1]);
			U[s2].zeros(ds.Nf[s2]);
			U[s1](p1) = 1;
			U[s2](p2) = 1;
			U[s1] = ds.Mi[s1] * U[s1];
			U[s2] = ds.Mi[s2] * U[s2];
			
			det[s1] = 1. + dot(V[s1], U[s1]);
			det[s2] = 1. + dot(V[s2], U[s2]);
			amp = abs_squared(det[0] * det[1]);
			if(amp > rng::uniform())
			{
				return_value = 4;
				accept = true;
				ds.Mi[s1] -= (U[s1] * V[s1] * ds.Mi[s1]) / det[s1];
				ds.Mi[s2] -= (U[s2] * V[s2] * ds.Mi[s2]) / det[s2];
			}
		}
		if(!step || accept)
		{
			ds.M[s1].row(p1) += V[s1];
			ds.M[s2].row(p2) += V[s2];
		}
	}
	
	if(!step || accept)
	{
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
		return return_value;
	}
	return 0;
}

bool apriori_swap_proposal(const arma::vec& w, const arma::uvec &Jo, const arma::uvec & Je, unsigned int &io, unsigned int &ie)
{
	double Zo1, Ze1, x, Zo2, Ze2, max_w;
	arma::vec Wo, We;
	
	max_w = max(w);
	Wo = w(Jo);
	We = w(Je);
	Wo = exp(0.5 * (Wo - max_w));
	We = exp(0.5 * (max_w - We));
	Wo = cumsum(Wo);
	We = cumsum(We);
	Zo1 = Wo(Wo.n_rows - 1);
	Ze1 = We(We.n_rows - 1);
	Wo /= Zo1;
	We /= Ze1;
	
	x = rng::uniform();
	io = 0;
	while(Wo(io) < x) io++;
	
	x = rng::uniform();
	ie = 0;
	while(We(ie) < x) ie++;
	
	Zo2 = Zo1 - exp(0.5 * (w(Jo(io)) - max_w)) + exp(0.5 * (w(Je(ie)) - max_w));
	Ze2 = Ze1 - exp(0.5 * (max_w - w(Je(ie)))) + exp(0.5 * (max_w - w(Jo(io))));
	
	return rng::uniform() < Zo1 * Ze1 / Zo2 / Ze2;	
}


template
void initial_configuration<double>(unsigned int Nu, unsigned int Nd, data_structures<double> &ds);

template
void initial_configuration<arma::cx_double>(unsigned int Nu, unsigned int Nd, data_structures<arma::cx_double> &ds);

template 
unsigned int rotate_face<double>(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	double &amp,
	data_structures<double> &ds);

template 
unsigned int rotate_face<arma::cx_double>(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	double &amp,
	data_structures<arma::cx_double> &ds);
