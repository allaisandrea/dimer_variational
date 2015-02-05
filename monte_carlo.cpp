#include <iomanip>
#include "monte_carlo.h"
#include "rng.h"
#include "linear_algebra.h"
#include "utilities.h"


template<class type>
void initial_configuration(unsigned int Nf, data_structures<type> &ds)
{
	unsigned int x, y, L, c, max_attempts;
	arma::uvec edges, e;
	ds.Nf = Nf;
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
	
	if(edges.n_elem < 2 * Nf)
		throw "Too many fermions";
	
	ds.particles.zeros(2 * L * L);
	ds.Mu.set_size(Nf, Nf);
	ds.Md.set_size(Nf, Nf);
	ds.J = sort_index(ds.w);
	ds.J.resize(Nf);
	for(c = 0; c < Nf; c++)
	{
		ds.particles(edges(c)) = c + 2;
		e << edges(c);
		ds.Mu.row(c) = ds.psi(e, ds.J);
	}
	for(; c < 2 * Nf; c++)
	{
		ds.particles(edges(c)) = c + 2;
		e << edges(c);
		ds.Md.row(c - Nf) = ds.psi(e, ds.J);
	}
	for(; c < edges.n_elem; c++)
	{
		ds.particles(edges(c)) = 1;
	}
	
	c = 0;
	max_attempts = 10;
	while(c < max_attempts && (singular(ds.Mu) || singular(ds.Md)))
	{
		while(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), false, ds) < 2);
		c++;
	}
	if(c == max_attempts)
		throw "Unable to find regular configuration";
	std::cout << "found regular configuration after " << c << " rotations.\n";
	
	ds.Mui = arma::inv(ds.Mu);
	ds.Mdi = arma::inv(ds.Md);
}

template <class type>
unsigned int rotate_face(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	data_structures<type> &ds)
{
	arma::uvec particles, edge;
	edge = ds.face_edges.col(f);
	particles = ds.particles(edge);
	if(particles(0) == 0 && particles(2) == 0)
	{
		if     (particles(1) == 1 && particles(3) == 1)
			return rotate_face_bb(edge(1), edge(2), edge(3), edge(0), step, ds);
		else if(particles(1) == 1 && particles(3) >= 2)
			if(clockwise)
				return rotate_face_bf(edge(1), edge(0), edge(3), edge(2), step, ds);
			else
				return rotate_face_bf(edge(1), edge(2), edge(3), edge(0), step, ds);
		else if(particles(1) >= 2 && particles(3) == 1)
			if(clockwise)
				return rotate_face_bf(edge(3), edge(2), edge(1), edge(0), step, ds);
			else
				return rotate_face_bf(edge(3), edge(0), edge(1), edge(2), step, ds);
		else if(particles(1) >= 2 && particles(3) >= 2)
			if(clockwise)
				return rotate_face_ff(edge(1), edge(0), edge(3), edge(2), step, ds);
			else
				return rotate_face_ff(edge(1), edge(2), edge(3), edge(0), step, ds);
	}
	else if(particles(1) == 0 && particles(3) == 0)
	{
		if     (particles(0) == 1 && particles(2) == 1)
			return rotate_face_bb(edge(0), edge(1), edge(2), edge(3), step, ds);
		else if(particles(0) == 1 && particles(2) >= 2)
			if(clockwise)
				return rotate_face_bf(edge(0), edge(3), edge(2), edge(1), step, ds);
			else
				return rotate_face_bf(edge(0), edge(1), edge(2), edge(3), step, ds);
		else if(particles(0) >= 2 && particles(2) == 1)
			if(clockwise)
				return rotate_face_bf(edge(2), edge(1), edge(0), edge(3), step, ds);
			else
				return rotate_face_bf(edge(2), edge(3), edge(0), edge(1), step, ds);
		else if(particles(0) >= 2 && particles(2) >= 2)
			if(clockwise)
				return rotate_face_ff(edge(0), edge(3), edge(2), edge(1), step, ds);
			else
				return rotate_face_ff(edge(0), edge(1), edge(2), edge(3), step, ds);
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
	data_structures<type> & ds)
{
	bool accept = false;
	double amp;
	
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
	data_structures<type> & ds)
{
	bool accept = false;
	unsigned int p;
	double amp;
	type det;
	arma::Mat<type> U, V, *M, *Mi;
	arma::uvec e;
	
	p = ds.particles(origin2) - 2;
	
	if(ds.particles(origin1) != 1 || p > 2 * ds.Nf)
		throw "logic error: this should never happen";
	
	if(p < ds.Nf)
	{
		M = &ds.Mu;
		Mi = &ds.Mui;
	}
	else
	{
		p -=  ds.Nf;
		M = &ds.Md;
		Mi = &ds.Mdi;
	}
	
	e << destination2;
	V = ds.psi(e, ds.J);
	e << origin2;
	V -= ds.psi(e, ds.J);
	
	if(step)
	{
		
		U.zeros(ds.Nf, 1);
		U(p, 0) = 1;
		U = (*Mi) * U;
		
		det = 1. + dot(V.row(0), U.col(0));
		amp = abs_squared(det * ds.phi(destination1) / ds.phi(origin1));
		if(amp > rng::uniform())
		{
// 			std::cout << std::setw(12) << amp;
			accept = true;
			(*Mi) -= (U * V * (*Mi)) / det;
		}

	}
	if(!step || accept)
	{
		M->row(p) += V;
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
	data_structures<type> & ds)
{
	bool accept = false;
	unsigned int p1, p2, isw, return_value;
	double amp;
	type det1, det2;
	arma::Mat<type> U, V, K, *M, *Mi;
	arma::uvec e;
	
	p1 = ds.particles(origin1) - 2;
	p2 = ds.particles(origin2) - 2;
	
	if(p1 > 2 * ds.Nf || p2 > 2 * ds.Nf)
		throw "logic error: this should never happen";
	
	if(p1 > p2)
	{
		swap(origin1, origin2);
		swap(destination1, destination2);
		swap(p1, p2);
	}
	
	if((p1 < ds.Nf && p2 < ds.Nf) || (p1 >= ds.Nf && p2 >= ds.Nf))
	{
		if(p1 < ds.Nf && p2 < ds.Nf)
		{
			M = &ds.Mu;
			Mi = &ds.Mui;
		}
		else if (p1 >= ds.Nf && p2 >= ds.Nf)
		{
			p1 -= ds.Nf;
			p2 -= ds.Nf;
			M = &ds.Md;
			Mi = &ds.Mdi;
		}
		else
			throw "logic error: this should never happen";
		
		e << destination1 << destination2;
		V = ds.psi(e, ds.J);
		e << origin1 << origin2;
		V -= ds.psi(e, ds.J);
			
		if(step)
		{

			U.zeros(ds.Nf, 2);
			U(p1, 0) = 1;
			U(p2, 1) = 1;
			U = (*Mi) * U;
			K = V * U;
			K(0, 0) += 1;
			K(1, 1) += 1;
			det1 = K(0, 0) * K(1, 1) - K(0, 1) * K(1, 0);
			amp = abs_squared(det1);
			if(amp > rng::uniform())
			{
// 				std::cout << std::setw(12) << amp;
				accept = true;
				(*Mi) -= U * inv(K) * V * (*Mi);
			}
		}
		if(!step || accept)
		{
			return_value = 3;
			M->row(p1) += V.row(0);
			M->row(p2) += V.row(1);
		}
	}
	else if((p1 < ds.Nf && p2 >= ds.Nf))
	{
		p2 -= ds.Nf;
		
		e << destination1 << destination2;
		V =  ds.psi(e, ds.J);
		e << origin1 << origin2;
		V -= ds.psi(e, ds.J);
			
		if(step)
		{
			
			U.zeros(ds.Nf, 2);
			U(p1, 0) = 1;
			U(p2, 1) = 1;
			U.col(0) = ds.Mui * U.col(0);
			U.col(1) = ds.Mdi * U.col(1);
			
			det1 = 1. + dot(V.row(0), U.col(0));
			det2 = 1. + dot(V.row(1), U.col(1));
			amp = abs_squared(det1 * det2);
			if(amp > rng::uniform())
			{
// 				std::cout << std::setw(12) << amp;
				return_value = 4;
				accept = true;
				ds.Mui -= (U.col(0) * V.row(0) * ds.Mui) / det1;
				ds.Mdi -= (U.col(1) * V.row(1) * ds.Mdi) / det2;
			}
		}
		if(!step || accept)
		{
			ds.Mu.row(p1) += V.row(0);
			ds.Md.row(p2) += V.row(1);
		}
	}
	else
		throw "logic error: this should never happen";
	
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

template
void initial_configuration<double>(unsigned int Nf, data_structures<double> &ds);

template
void initial_configuration<arma::cx_double>(unsigned int Nf, data_structures<arma::cx_double> &ds);

template 
unsigned int rotate_face<double>(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	data_structures<double> &ds);

template 
unsigned int rotate_face<arma::cx_double>(
	unsigned int f, 
	unsigned int clockwise, 
	bool step,
	data_structures<arma::cx_double> &ds);
