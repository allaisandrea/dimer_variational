#include <exception.h>
#include "data_structures.h"

template<class type>
void initial_configuration(unsigned int Nf, data_structures<type> &ds)
{
	unsigned int x, y, L, c;
	arma::ivec edges, e;
	ds.Nf = Nf;
	L = ds.L;
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
		throw logic_error;
	
	ds.particles.zeros(2 * L * L);
	ds.Mu.set_size(Nf, Nf);
	ds.Md.set_size(Nf, Nf);
	ds.J = sort_index(ds.w);
	ds.J.resize(Nf);
	for(c = 0; c < Nf; c++)
	{
		ds.particles(edges(c)) = c + 2;
		ds.Mu.row(c) = ds.psi(edges.rows(c,c), ds.J);
		
	}
	for(; c < 2 * Nf; c++)
	{
		ds.particles(edges(c)) = c + 2;
		ds.Md.row(c - Nf) = ds.psi(edges.rows(c,c), ds.J);
	}
	for(; c < edges.n_elem; c++)
	{
		ds.particles(edges(c)) = 1;
	}
	
	unsigned int f, clockwise;
	while(singular(ds.Mu) || singular(ds.Md))
	{
		f = rng::uniform_integer(ds.n_faces);
		clockwise = rng::uniform_integer(2);
		rotate_face(f, clockwise, false, ds);
	}
}

template <class type>
void rotate_face(
	unsigned int f, 
	bool clockwise, 
	bool step,
	data_structures<type> &ds)
{
	arma::uvec particles, edge;
	edge = ds.face_edges.col(f);
	particles = ds.particles(edge);
	if(particles(0) == 0 || particles(2) == 0)
	{

		if     (particles(1) == 1 && particles(3) == 1)
			rotate_face_bb(edge(1), edge(2), edge(3), edge(0), step, ds);
		else if(particles(1) == 1 && particles(3) >= 2)
			if(clockwise)
				rotate_face_bf(edge(1), edge(0), edge(3), edge(2), step, ds);
			else
				rotate_face_bf(edge(1), edge(2), edge(3), edge(0), step, ds);
		else if(particles(1) >= 2 && particles(3) == 1)
			if(clockwise)
				rotate_face_bf(edge(3), edge(2), edge(1), edge(0), step, ds);
			else
				rotate_face_bf(edge(3), edge(0), edge(1), edge(2), step, ds);
		else if(particles(1) >= 2 && particles(3) >= 2)
			if(clockwise)
				rotate_face_ff(edge(1), edge(0), edge(3), edge(2), step, ds);
			else
				rotate_face_ff(edge(1), edge(2), edge(3), edge(0), step, ds);
		else
			throw runtime_error;
	}
	else if(particles(1) == 0 && particles(3) == 0)
	{
		if     (particles(0) == 1 && particles(2) == 1)
			rotate_face_bb(edge(0), edge(1), edge(2), edge(3), step, ds);
		else if(particles(0) == 1 && particles(2) >= 2)
			if(clockwise)
				rotate_face_bf(edge(0), edge(3), edge(2), edge(1), step, ds);
			else
				rotate_face_bf(edge(0), edge(1), edge(2), edge(3), step, ds);
		else if(particles(0) >= 2 && particles(2) == 1)
			if(clockwise)
				rotate_face_bf(edge(2), edge(1), edge(0), edge(3), step, ds);
			else
				rotate_face_bf(edge(2), edge(3), edge(0), edge(1), step, ds);
		else if(particles(0) >= 2 && particles(2) >= 2)
			if(clockwise)
				rotate_face_ff(edge(0), edge(3), edge(2), edge(1), step, ds);
			else
				rotate_face_ff(edge(0), edge(1), edge(2), edge(3), step, ds);
		else
			throw runtime_error;
	}
	else
		throw runtime_error;
}


template<class type>
void rotate_face_bb(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	data_structures<type> & ds)
{
	bool accept;
	
	if(ds.particles(origin1) != 1 || ds.particles(origin2) != 1)
		throw logic_error;
	
	
	if(step)
	{
		if(abs_squared(ds.phi(destination1) * ds.phi(destination2) / ds.phi(origin1) / ds.phi(origin2)) > rng.random_uniform())
		{
			accept = true;
		}
	}
	if(!step || accept)
	{
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
	}
	
}

template<class type>
void rotate_face_bf(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	data_structures<type> & ds)
{
	bool accept;
	unsigned int p;
	type det;
	arma::Mat<type> U, V, *M, *Mi;
	arma::ivec e;
	
	p = ds.particles(origin2) - 2;
	
	if(ds.particles(e01) != 1 || p > 2 * ds.Nf)
		throw logic_error;
	
	if(p < ds.Nf)
	{
		M = &ds.Mu;
		Mi = &ds.Mui;
	}
	else
	{
		p -=  ds.Nf
		M = &ds.Md;
		Mi = &ds.Mdi;
	}
		
	if(step)
	{
		e << destination2;
		V = psi(e, ds.J);
		e << origin2;
		V -= psi(e, ds.J);
		U.zeros(ds.Nf, 1);
		U(p, 0) = 1;
		U = (*Mi) * U;
		
		det = 1. + V * U;
		
		if(abs_squared(det * ds.phi(destination1) / ds.phi(origin1)) > rng.random_uniform())
		{
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
	}
	
}


template<class type>
void rotate_face_ff(
	unsigned int origin1, 
	unsigned int destination1,
	unsigned int origin2,
	unsigned int destination2,
	bool step,
	data_structures<type> & ds)
{
	bool accept;
	unsigned int p1, p2, isw;
	type det1, det2;
	arma::Mat<type> U, V, K, *M, *Mi;
	arma::ivec e;
	p1 = ds.particles(origin1) - 2;
	p2 = ds.particles(origin2) - 2;
	
	if(p1 > 2 * ds.Nf || p2 > 2 * ds.Nf)
		throw logic_error;
	
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
			throw logic_error;
		
		if(step)
		{
			e << destination1 << destination2;
			V = psi(e, ds.J);
			e << origin1 << origin2;
			V -= psi(e, ds.J);
			U.zeros(ds.Nf, 2);
			U(p1, 0) = 1;
			U(p2, 1) = 1;
			U = (*Mi) * U;
			K = V * U;
			K(0, 0) += 1;
			K(1, 1) += 1;
			det1 = K(0, 0) * K(1, 1) - K(0, 1) * K(1, 0);
			
			if(abs_squared(det1) > rng.random_uniform())
			{
				accept = true;
				(*Mi) -= U * inv(K) * V * (*Mi);
			}
		}
		if(!step || accept)
		{
			M->row(p1) += V.row(0);
			M->row(p2) += V.row(1);
		}
	}
	else if((p1 < ds.Nf && p2 >= ds.Nf))
	{
		p2 -= ds.Nf;
		if(step)
		{
			e << destination1 << destination2;
			V = psi(e, ds.J);
			e << origin1 << origin2;
			V -= psi(e, ds.J);
			U.zeros(ds.Nf, 2);
			U(p1, 0) = 1;
			U(p2, 1) = 1;
			U.col(0) = ds.Mui * U.col(0);
			U.col(1) = ds.Mud * U.col(1);
			
			det1 = 1 + V.row(0) * U.col(0);
			det2 = 1 + V.row(1) * U.col(1);
			
			if(abs_squared(det1 * det2) > rng.random_uniform())
			{
				accept = true;
				ds.Mui -= (U * V * ds.Mui) / det1;
				ds.Mdi -= (U * V * ds.Mdi) / det2;
			}
		}
		if(!step || accept)
		{
			Mu.row(p1) += V.row(0);
			Md.row(p2) += V.row(1);
		}
	}
	else
		throw logic_error;
	
	if(!step || accept)
	{
		ds.particles(destination1) = ds.particles(origin1);
		ds.particles(destination2) = ds.particles(origin2);
		ds.particles(origin1) = 0;
		ds.particles(origin2) = 0;
	}
	
}

template
void initial_configuration<double>(unsigned int Nf, data_structures<double> &ds);

template
void initial_configuration<arma::cx_double>(unsigned int Nf, data_structures<arma::cx_double> &ds);

template 
void rotate_face<double>(
	unsigned int f, 
	bool clockwise, 
	bool step,
	data_structures<double> &ds);

template 
void rotate_face<arma::cx_double>(
	unsigned int f, 
	bool clockwise, 
	bool step,
	data_structures<arma::cx_double> &ds);
