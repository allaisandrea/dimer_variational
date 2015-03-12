#include <iomanip>
#include <exception>
#include "monte_carlo.h"
#include "rng.h"
#include "linear_algebra.h"
#include "utilities.h"

// #define lock_face_bb
// #define lock_face_bf
// #define lock_face_ff

template<class type>
void initial_configuration(data_structures<type> &ds)
{
	unsigned int x, y, L, c, s, max_attempts, Nu, Nd;
	double dummy;
	arma::uvec edges, e;
	
	Nu = ds.Nf[0];
	Nd = ds.Nf[1];
	L = ds.L;
	
	if(L % 2 == 1)
		throw std::logic_error("Lattice side must be even");
	
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
		throw std::logic_error("Too many fermions");
	
	ds.particles.zeros(2 * L * L);
	ds.M[0].set_size(Nu, Nu);
	ds.M[1].set_size(Nd, Nd);
	
	for(s = 0; s < 2; s++)
	{
		ds.J[s] = sort_index(ds.w[s]);
		ds.K[s] = ds.J[s].rows(ds.Nf[s], ds.J[s].n_rows - 1);
		ds.J[s].resize(ds.Nf[s]);
	}
	
	ds.fermion_edge[0].set_size(Nu);
	ds.fermion_edge[1].set_size(Nd);
	for(c = 0; c < Nu + Nd; c++)
	{
		ds.particles(edges(c)) = c + 2;
		e << edges(c);
		c < Nu ? s = 0 : s = 1;
		ds.M[s].row(c - s * Nu) = ds.psi[s](e, ds.J[s]);
		ds.fermion_edge[s](c - s * Nu) = edges(c);
	}
	
	for(c = Nu + Nd; c < edges.n_elem; c++)
	{
		ds.particles(edges(c)) = 1;
		ds.boson_edges.insert(edges(c));
	}
	
	c = 0;
	max_attempts = 1000;
	while(c < max_attempts && ((ds.M[0].n_rows > 0 && singular(ds.M[0])) || (ds.M[1].n_rows > 0 && singular(ds.M[1]))))
	{
		while(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), false, dummy, ds) < 2); 
		c++;
	}
	if(c == max_attempts)
		throw std::runtime_error("Unable to find regular configuration");
	std::cout << "found regular configuration after " << c << " rotations.\n";
	for(s = 0; s < 2; s++)
		if(ds.M[s].n_rows > 0)
			ds.Mi[s] = arma::inv(ds.M[s]);
}

template <class type>
unsigned int rotate_face(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	double &amp,
	data_structures<type> &ds)
{
	static unsigned int particles[4], edge[4];
	
	copy_line((unsigned int) 0, (unsigned int) 1, 4, 1, ds.face_edges.memptr() + 4 * f, 1, edge);
	copy_line_sparse((unsigned int) 0, (unsigned int) 1, 4, edge, 1, ds.particles.memptr(), 1, particles);
	
	if(particles[0] == 0 && particles[2] == 0)
	{
		if     (particles[1] == 1 && particles[3] == 1)
			return rotate_face_bb(edge[1], edge[2], edge[3], edge[0], step, amp, ds);
		else if(particles[1] == 1 && particles[3] >= 2)
			if(clockwise)
				return rotate_face_bf(edge[1], edge[0], edge[3], edge[2], step, amp, ds);
			else
				return rotate_face_bf(edge[1], edge[2], edge[3], edge[0], step, amp, ds);
		else if(particles[1] >= 2 && particles[3] == 1)
			if(clockwise)
				return rotate_face_bf(edge[3], edge[2], edge[1], edge[0], step, amp, ds);
			else
				return rotate_face_bf(edge[3], edge[0], edge[1], edge[2], step, amp, ds);
		else if(particles[1] >= 2 && particles[3] >= 2)
			if(clockwise)
				return rotate_face_ff(edge[1], edge[0], edge[3], edge[2], step, amp, ds);
			else
				return rotate_face_ff(edge[1], edge[2], edge[3], edge[0], step, amp, ds);
	}
	else if(particles[1] == 0 && particles[3] == 0)
	{
		if     (particles[0] == 1 && particles[2] == 1)
			return rotate_face_bb(edge[0], edge[1], edge[2], edge[3], step, amp, ds);
		else if(particles[0] == 1 && particles[2] >= 2)
			if(clockwise)
				return rotate_face_bf(edge[0], edge[3], edge[2], edge[1], step, amp, ds);
			else
				return rotate_face_bf(edge[0], edge[1], edge[2], edge[3], step, amp, ds);
		else if(particles[0] >= 2 && particles[2] == 1)
			if(clockwise)
				return rotate_face_bf(edge[2], edge[1], edge[0], edge[3], step, amp, ds);
			else
				return rotate_face_bf(edge[2], edge[3], edge[0], edge[1], step, amp, ds);
		else if(particles[0] >= 2 && particles[2] >= 2)
			if(clockwise)
				return rotate_face_ff(edge[0], edge[3], edge[2], edge[1], step, amp, ds);
			else
				return rotate_face_ff(edge[0], edge[1], edge[2], edge[3], step, amp, ds);
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
		throw std::logic_error("This should never happen");
	
	
	if(step)
	{
		amp = abs_squared(ds.phi(destination1) * ds.phi(destination2) / ds.phi(origin1) / ds.phi(origin2));
		if(amp > rng::uniform())
			accept = true;
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
	static arma::Col<type> U[2];
	static arma::Row<type> V[2], buf;
	
	p = ds.particles(origin2) - 2;
	
	if(ds.particles(origin1) != 1 || p > ds.Nf[0] + ds.Nf[1])
		throw std::logic_error("This should never happen");
	
	if(p < ds.Nf[0])
		s = 0;
	else
	{
		p -=  ds.Nf[0];
		s = 1;
	}
	
	V[s].set_size(ds.Nf[s]);
	copy_line_sparse((type) 0., (type) +1., ds.Nf[s], ds.J[s].memptr(), ds.psi[s].n_rows, ds.psi[s].memptr() + destination2, 1, V[s].memptr());
	copy_line_sparse((type) 1., (type) -1., ds.Nf[s], ds.J[s].memptr(), ds.psi[s].n_rows, ds.psi[s].memptr() + origin2, 1, V[s].memptr());
	
	if(step)
	{
		U[s].set_size(ds.Nf[s]);
		copy_line((type) 0., (type) 1., ds.Nf[s], 1, ds.Mi[s].colptr(p), 1, U[s].memptr());
		
		det = 1. + dot(V[s], U[s]);
		amp = abs_squared(det * ds.phi(destination1) / ds.phi(origin1));
		if(amp > rng::uniform())
		{
			accept = true;
			buf = V[s] * ds.Mi[s];
			rank_1_update(-1. / det, U[s], buf, ds.Mi[s]);
		}

	}
	if(!step || accept)
	{
		copy_line((type) 1., (type) 1., ds.Nf[s], 1, V[s].memptr(), ds.Nf[s], ds.M[s].memptr() + p);
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
		ds.fermion_edge[s](p) = destination2;
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
	static arma::Mat<type> U2[2], V2[2], K, buf;
	static arma::Row<type> V[2];
	static arma::Col<type> U[2];
	
	p1 = ds.particles(origin1) - 2;
	p2 = ds.particles(origin2) - 2;
	
	if(p1 > ds.Nf[0] + ds.Nf[1])
		throw std::logic_error("This should never happen");
	if(p2 > ds.Nf[0] + ds.Nf[1])
		throw std::logic_error("This should never happen");
	
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
		V2[s1].set_size(2, ds.Nf[s1]);
		
		copy_line_sparse((type) 0., (type) +1., ds.Nf[s1], ds.J[s1].memptr(), ds.psi[s1].n_rows, ds.psi[s1].memptr() + destination1, 2, V2[s1].memptr());
		copy_line_sparse((type) 1., (type) -1., ds.Nf[s1], ds.J[s1].memptr(), ds.psi[s1].n_rows, ds.psi[s1].memptr() + origin1,      2, V2[s1].memptr());
		
		copy_line_sparse((type) 0., (type) +1., ds.Nf[s1], ds.J[s1].memptr(), ds.psi[s1].n_rows, ds.psi[s1].memptr() + destination2, 2, V2[s1].memptr() + 1);
		copy_line_sparse((type) 1., (type) -1., ds.Nf[s1], ds.J[s1].memptr(), ds.psi[s1].n_rows, ds.psi[s1].memptr() + origin2     , 2, V2[s1].memptr() + 1);
				
		if(step)
		{
			U2[s1].set_size(ds.Nf[s1], 2);
			copy_line((type) 0., (type) 1., ds.Nf[s1], 1, ds.Mi[s1].colptr(p1), 1, U2[s1].colptr(0));
			copy_line((type) 0., (type) 1., ds.Nf[s1], 1, ds.Mi[s1].colptr(p2), 1, U2[s1].colptr(1));
			
			K = V2[s1] * U2[s1];
			K[0] += 1.;
			K[3] += 1.;
			det[s1] = K[0] * K[3] - K[1] * K[2];
			amp = abs_squared(det[s1]);
			if(amp > rng::uniform())
			{
				accept = true;
				buf = V2[s1] * ds.Mi[s1];
				buf = inv(K) * buf;
				rank_k_update((type)-1., U2[s1], buf, ds.Mi[s1]);
			}
		}
		if(!step || accept)
		{
			return_value = 3;
			copy_line((type) 1., (type) 1., ds.Nf[s1], 2, V2[s1].memptr() + 0, ds.Nf[s1], ds.M[s1].memptr() + p1);
			copy_line((type) 1., (type) 1., ds.Nf[s1], 2, V2[s1].memptr() + 1, ds.Nf[s1], ds.M[s1].memptr() + p2);
		}
	}
	else
	{
		V[s1].set_size(ds.Nf[s1]);
		copy_line_sparse((type) 0., (type) +1., ds.Nf[s1], ds.J[s1].memptr(), ds.psi[s1].n_rows, ds.psi[s1].memptr() + destination1, 1, V[s1].memptr());
		copy_line_sparse((type) 1., (type) -1., ds.Nf[s1], ds.J[s1].memptr(), ds.psi[s1].n_rows, ds.psi[s1].memptr() + origin1,      1, V[s1].memptr());
		
		V[s2].set_size(ds.Nf[s2]);
		copy_line_sparse((type) 0., (type) +1., ds.Nf[s2], ds.J[s2].memptr(), ds.psi[s2].n_rows, ds.psi[s2].memptr() + destination2, 1, V[s2].memptr());
		copy_line_sparse((type) 1., (type) -1., ds.Nf[s2], ds.J[s2].memptr(), ds.psi[s2].n_rows, ds.psi[s2].memptr() + origin2,      1, V[s2].memptr());
		
		if(step)
		{
			U[s1].set_size(ds.Nf[s1]);
			copy_line((type) 0., (type) 1., ds.Nf[s1], 1, ds.Mi[s1].colptr(p1), 1, U[s1].memptr());
			U[s2].set_size(ds.Nf[s2]);
			copy_line((type) 0., (type) 1., ds.Nf[s2], 1, ds.Mi[s2].colptr(p2), 1, U[s2].memptr());
			
			det[s1] = 1. + dot(V[s1], U[s1]);
			det[s2] = 1. + dot(V[s2], U[s2]);
			amp = abs_squared(det[0] * det[1]);
			if(amp > rng::uniform())
			{
				return_value = 4;
				accept = true;
				buf = V[s1] * ds.Mi[s1];
				rank_k_update(-1. / det[s1], U[s1], buf, ds.Mi[s1]);
				buf = V[s2] * ds.Mi[s2];
				rank_k_update(-1. / det[s2], U[s2], buf, ds.Mi[s2]);
			}
		}
		if(!step || accept)
		{
			return_value = 4;
			copy_line((type) 1., (type) 1., ds.Nf[s1], 1, V[s1].memptr(), ds.Nf[s1], ds.M[s1].memptr() + p1);
			copy_line((type) 1., (type) 1., ds.Nf[s2], 1, V[s2].memptr(), ds.Nf[s2], ds.M[s2].memptr() + p2);
		}
	}
	
	if(!step || accept)
	{
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
		ds.fermion_edge[s1](p1) = destination1;
		ds.fermion_edge[s2](p2) = destination2;
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

template <class type>
bool swap_states(unsigned int s, double& amp, data_structures<type> & ds)
{
	unsigned int io, ie;
	type det;
	arma::uvec e;
	arma::Col<type> U, buf;
	arma::Row<type> V;
	

	if(ds.Nf[s] > 0 && apriori_swap_proposal(ds.w[s], ds.J[s], ds.K[s], io, ie))
	{
		
		
		e << ds.K[s](ie);
		U = ds.psi[s](ds.fermion_edge[s], e);
		
		e << ds.J[s](io);
		U -= ds.psi[s](ds.fermion_edge[s], e);
		
		V = ds.Mi[s].row(io);
		
		det = 1. + dot(V, U);
		
		amp = abs_squared(det);
		if(amp > rng::uniform())
		{
			ds.M[s].col(io) += U;
			buf = ds.Mi[s] * U;
			rank_k_update(-1. / det, buf, V, ds.Mi[s]);
			swap(ds.J[s](io), ds.K[s](ie));
			return true;
		}
	}
	
	return false;
}

template
void initial_configuration<double>(data_structures<double> &ds);

template
void initial_configuration<arma::cx_double>(data_structures<arma::cx_double> &ds);

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

template 
bool swap_states<double>(unsigned int s, double& amp, data_structures<double> & ds);

template 
bool swap_states<arma::cx_double>(unsigned int s, double& amp, data_structures<arma::cx_double> & ds);