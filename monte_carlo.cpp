#include <iomanip>
#include <exception>
#include "monte_carlo.h"
#include "rng.h"
#include "linear_algebra.h"
#include "utilities.h"

template<class type>
unsigned int rotate_face_bb(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	double &amp,
	data_structures<type> & ds);

template<class type>
unsigned int rotate_face_bf(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	double & amp,
	data_structures<type> & ds);

template<class type>
bool apriori_swap_proposal(unsigned int s, const data_structures<type> &ds, unsigned int &io, double &Zo2, unsigned int &ie, double &Ze2);

template<class type>
void initial_configuration(data_structures<type> &ds)
{
	unsigned int x, y, L, c, i, s, max_attempts;
	double dummy, w1, w2;
	arma::uvec edges;
	
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
	
	if(edges.n_elem < ds.Nf[0] + ds.Nf[1])
		throw std::logic_error("Too many fermions");
	
	ds.particles.zeros(2 * L * L);
	
	
	c = 0;
	for(s = 0; s < 2; s++)
	{
		if(ds.Nf[s] > 0)
		{
			ds.J[s] = sort_index(ds.w[s]);
			w1 = ds.w[s](ds.J[s](ds.Nf[s] - 1));
			w2 = ds.w[s](ds.J[s](ds.Nf[s]    ));
			ds.Epw[s] = exp(0.5 * (ds.w[s] -  w1));
			ds.Emw[s] = exp(0.5 * (w2 - ds.w[s]));
			ds.Epw[s] = ds.Epw[s](ds.J[s]);
			ds.Emw[s] = ds.Emw[s](ds.J[s]);
			ds.Zo[s] = accu(ds.Epw[s].rows(0, ds.Nf[s] - 1));
			ds.Ze[s] = accu(ds.Emw[s].rows(ds.Nf[s], ds.Emw[s].n_rows - 1));
			
			ds.M[s].set_size(ds.Nf[s], ds.Nf[s]);
			ds.fermion_edge[s].set_size(ds.Nf[s]);
			for(i = 0; i < ds.Nf[s]; i++)
			{
				ds.particles(edges(c)) = c + 2;
				ds.M[s].row(i) = ds.psi[s](edges.rows(c, c), ds.J[s].rows(0, ds.Nf[s] - 1));
				ds.fermion_edge[s](i) = edges(c);
				c++;
			}
		}
	}
	
	
	for(c = ds.Nf[0] + ds.Nf[1]; c < edges.n_elem; c++)
		ds.particles(edges(c)) = 1;
	
	
	c = 0;
	max_attempts = 1000;
	while(c < max_attempts && ((ds.M[0].n_rows > 0 && singular(ds.M[0])) || (ds.M[1].n_rows > 0 && singular(ds.M[1]))))
	{
		while(monte_carlo_step(false, false, dummy, ds) < 2); 
		c++;
	}
	if(c == max_attempts)
		throw std::runtime_error("Unable to find regular configuration");
	
	for(s = 0; s < 2; s++)
		if(ds.M[s].n_rows > 0)
			ds.Mi[s] = arma::inv(ds.M[s]);
}

template
void initial_configuration<double>(data_structures<double> &ds);

template
void initial_configuration<arma::cx_double>(data_structures<arma::cx_double> &ds);


template <class type>
unsigned int monte_carlo_step(
	bool step,
	bool swap_states,
	double &amp,
	data_structures<type> &ds)
{
	unsigned int f, clockwise, io, ie, i, i1, i2, i3, p, s;
	static unsigned int particles[4], edge[4];
	bool accept;
	double Zo, Ze;
	type det;
	static arma::Mat<type> V[2], U[2], K, buf;

	
	f = rng::uniform_integer(ds.n_faces);
	clockwise = rng::uniform_integer(2);
	
	for(i = 0; i < 4; i++)
		edge[i] = ds.face_edges[4 * f + i];
	copy_vector_sparse(4, edge, ds.particles.memptr(), 1, particles, 1);
	
	i = 0;
	while(i < 4 && !ds.is_boson(particles[i])) i++;
	
	i2 = (i + 2) % 4;
	if(i == 4 || ds.is_empty(particles[i2]))
		return 0;
	
	i1 = (i + 1) % 4;
	i3 = (i + 3) % 4;
		
	if(ds.is_boson(particles[i2]))
	{
		return rotate_face_bb(edge[i], edge[i1], edge[i2], edge[i3], step, amp, ds);
	}
	else
	{
		if(clockwise) swap(i1, i3);
		ds.to_p_s(particles[i2], p, s);
		accept = swap_states && apriori_swap_proposal(s, ds, io, Zo, ie, Ze);
		if(accept)
		{
			accept = false;
			if(step)
			{
				V[s].set_size(2, ds.M[s].n_cols);
				copy_vector_sparse(ds.M[s].n_cols, ds.J[s].memptr(), ds.psi[s].memptr() + edge[i3], ds.psi[s].n_rows, V[s].memptr(), V[s].n_rows);
				add_to_vector(ds.M[s].n_cols, (type) -1., ds.M[s].memptr() + p, ds.M[s].n_rows, V[s].memptr(), V[s].n_rows);
				
				V[s].row(1).zeros();
				V[s](1, io) = 1.;
				V[s](0, io) = 0.;
				
				
				U[s].set_size(ds.M[s].n_rows, 2);
				copy_vector(ds.Mi[s].n_rows, ds.Mi[s].colptr(p), 1, U[s].colptr(0), 1);
				
				copy_vector_sparse(ds.M[s].n_rows, ds.fermion_edge[s].memptr(), ds.psi[s].colptr(ds.J[s](ie)), 1, U[s].colptr(1), 1);
				add_to_vector(ds.M[s].n_rows, (type) -1., ds.M[s].colptr(io), 1,  U[s].colptr(1), 1);
				U[s](p, 1) = ds.psi[s](edge[i3], ds.J[s](ie)) - ds.M[s](p, io);
				U[s].col(1) = ds.Mi[s] * U[s].col(1);
				
				K = V[s] * U[s];
				K[0] += 1.;
				K[3] += 1.;
				det = K[0] * K[3] - K[1] * K[2];
				
				amp = abs_squared(det);
				if(amp > rng::uniform())
				{
					accept = true;
					buf = V[s] * ds.Mi[s];
					buf = inv(K) * buf;
					rank_k_update((type)-1., U[s], buf, ds.Mi[s]);
				}
			}
			
			if(!step || accept)
			{
				ds.particles(edge[i1]) = ds.particles(edge[i]);
				ds.particles(edge[i3]) = ds.particles(edge[i2]);
				ds.particles(edge[i]) = 0;
				ds.particles(edge[i2]) = 0;
				ds.fermion_edge[s](p) = edge[i3];
				
				swap(ds.J[s](io), ds.J[s](ie));
				swap(ds.Epw[s](io), ds.Epw[s](ie));
				swap(ds.Emw[s](io), ds.Emw[s](ie));
				ds.Zo[s] = Zo;
				ds.Ze[s] = Ze;
			
				copy_vector_sparse(ds.M[s].n_cols, ds.J[s].memptr(), ds.psi[s].memptr() + edge[i3], ds.psi[s].n_rows, ds.M[s].memptr() + p, ds.M[s].n_rows);
				copy_vector_sparse(ds.M[s].n_rows, ds.fermion_edge[s].memptr(), ds.psi[s].colptr(ds.J[s](io)), 1, ds.M[s].colptr(io), 1);
				return 5;
			}
		}
		else
			return rotate_face_bf(edge[i], edge[i1], edge[i2], edge[i3], step, amp, ds);
	}
	
	return 0;
}

template
unsigned int monte_carlo_step<double>(
	bool step,
	bool swap_states,
	double &amp,
	data_structures<double> &ds);

template
unsigned int monte_carlo_step<arma::cx_double>(
	bool step,
	bool swap_states,
	double &amp,
	data_structures<arma::cx_double> &ds);

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
	
	V[s].set_size(ds.M[s].n_cols);
	copy_vector_sparse(ds.M[s].n_cols, ds.J[s].memptr(), ds.psi[s].memptr() + destination2, ds.psi[s].n_rows, V[s].memptr(), V[s].n_rows);
	add_to_vector(ds.M[s].n_cols, (type) -1., ds.M[s].memptr() + p, ds.M[s].n_rows, V[s].memptr(), V[s].n_rows);
	
	if(step)
	{
		U[s].set_size(ds.M[s].n_rows);
		copy_vector(ds.Mi[s].n_rows, ds.Mi[s].colptr(p), 1, U[s].memptr(), 1);
		
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
		add_to_vector(V[s].n_cols, (type) 1., V[s].memptr(), V[s].n_rows, ds.M[s].memptr() + p, ds.M[s].n_rows);
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
bool apriori_swap_proposal(unsigned int s, const data_structures<type> &ds, unsigned int &io, double &Zo2, unsigned int &ie, double &Ze2)
{
	unsigned int N, Nm;
	double x;
	const double* Epw, *Emw;
	N = ds.Nf[s];
	Nm = ds.Emw[s].n_elem;
	Epw = ds.Epw[s].memptr();
	Emw = ds.Emw[s].memptr();
	
	
	x = ds.Zo[s] * rng::uniform();
	io = 0;
	while(x > 0 && io < N)
	{
		x -= Epw[io];
		io++;
	}
	if(io > 0) io--;
	
	
	x = ds.Ze[s] * rng::uniform();
	ie = N;
	while(x > 0 && ie < Nm)
	{
		x -= Emw[ie];
		ie++;
	}
	if(ie > 0) ie--;
	
	Zo2 = ds.Zo[s] - Epw[io] + Epw[ie];
	Ze2 = ds.Ze[s] - Emw[ie] + Emw[io];
	
	
	x = ds.Zo[s] * ds.Ze[s] / Zo2 / Ze2;
	
	return rng::uniform() < x;
}

