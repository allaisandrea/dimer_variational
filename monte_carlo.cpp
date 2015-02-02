#include <exception.h>
#include "data_structures.h"

template<class type>
void rotate_face_bb(
	unsigned int eo1, 
	unsigned int ed1,
	unsigned int eo2,
	unsigned int ed2,
	bool step,
	data_structures<type> & ds)
{
	bool accept;
	
	if(ds.P(eo1) != 1 || ds.P(eo2) != 1)
		throw logic_error;
	
	
	if(step)
	{
		if(abs_squared(ds.phi(ed1) * ds.phi(ed2) / ds.phi(eo1) / ds.phi(eo2)) > rng.random_uniform())
		{
			accept = true;
		}
	}
	if(!step || accept)
	{
		ds.P(ed1) = ds.P(eo1);
		ds.P(ed2) = ds.P(eo2);
		ds.P(eo1) = 0;
		ds.P(eo2) = 0;
	}
	
}

template<class type>
void rotate_face_bf(
	unsigned int eo1, 
	unsigned int ed1,
	unsigned int eo2,
	unsigned int ed2,
	bool step,
	data_structures<type> & ds)
{
	bool accept;
	unsigned int p;
	type det;
	arma::Mat<type> U, V, *M, *Mi;
	arma::ivec e;
	
	p = ds.P(eo2) - 2;
	
	if(ds.P(e01) != 1 || p > 2 * ds.Nf)
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
		e << ed2;
		V = psi(e, ds.J);
		e << eo2;
		V -= psi(e, ds.J);
		U.zeros(ds.Nf, 1);
		U(p, 0) = 1;
		U = (*Mi) * U;
		
		det = 1. + V * U;
		
		if(abs_squared(det * ds.phi(ed1) / ds.phi(eo1)) > rng.random_uniform())
		{
			accept = true;
			(*Mi) -= (U * V * (*Mi)) / det;
		}
	}
	if(!step || accept)
	{
		M->row(p) += V;
		ds.P(ed1) = ds.P(eo1);
		ds.P(ed2) = ds.P(eo2);
		ds.P(eo1) = 0;
		ds.P(eo2) = 0;
	}
	
}


template<class type>
void rotate_face_ff(
	unsigned int eo1, 
	unsigned int ed1,
	unsigned int eo2,
	unsigned int ed2,
	bool step,
	data_structures<type> & ds)
{
	bool accept;
	unsigned int p1, p2, isw;
	type det1, det2;
	arma::Mat<type> U, V, K, *M, *Mi;
	arma::ivec e;
	p1 = ds.P(eo1) - 2;
	p2 = ds.P(eo2) - 2;
	
	if(p1 > 2 * ds.Nf || p2 > 2 * ds.Nf)
		throw logic_error;
	
	if(p1 > p2)
	{
		swap(eo1, eo2);
		swap(ed1, ed2);
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
			e << ed1 << ed2;
			V = psi(e, ds.J);
			e << eo1 << eo2;
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
			e << ed1 << ed2;
			V = psi(e, ds.J);
			e << eo1 << eo2;
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
		ds.P(ed1) = ds.P(eo1);
		ds.P(ed2) = ds.P(eo2);
		ds.P(eo1) = 0;
		ds.P(eo2) = 0;
	}
	
}

template <class type>
void rotate_face(
	unsigned int f, 
	bool clockwise, 
	bool step,
	data_structures<type> &ds)
{
	arma::uvec P, E;
	E = ds.FE(f);
	P = ds.P(E);
	if(P(0) == 0 || P(2) == 0)
	{

		if     (P(1) == 1 && P(3) == 1)
			rotate_face_bb(E(1), E(2), E(3), E(0), step, ds);
		else if(P(1) == 1 && P(3) >= 2)
			if(clockwise)
				rotate_face_bf(E(1), E(0), E(3), E(2), step, ds);
			else
				rotate_face_bf(E(1), E(2), E(3), E(0), step, ds);
		else if(P(1) >= 2 && P(3) == 1)
			if(clockwise)
				rotate_face_bf(E(3), E(2), E(1), E(0), step, ds);
			else
				rotate_face_bf(E(3), E(0), E(1), E(2), step, ds);
		else if(P(1) >= 2 && P(3) >= 2)
			if(clockwise)
				rotate_face_ff(E(1), E(0), E(3), E(2), step, ds);
			else
				rotate_face_ff(E(1), E(2), E(3), E(0), step, ds);
		else
			throw runtime_error;
	}
	else if(P(1) == 0 && P(3) == 0)
	{
		if     (P(0) == 1 && P(2) == 1)
			rotate_face_bb(E(0), E(1), E(2), E(3), step, ds);
		else if(P(0) == 1 && P(2) >= 2)
			if(clockwise)
				rotate_face_bf(E(0), E(3), E(2), E(1), step, ds);
			else
				rotate_face_bf(E(0), E(1), E(2), E(3), step, ds);
		else if(P(0) >= 2 && P(2) == 1)
			if(clockwise)
				rotate_face_bf(E(2), E(1), E(0), E(3), step, ds);
			else
				rotate_face_bf(E(2), E(3), E(0), E(1), step, ds);
		else if(P(0) >= 2 && P(2) >= 2)
			if(clockwise)
				rotate_face_ff(E(0), E(3), E(2), E(1), step, ds);
			else
				rotate_face_ff(E(0), E(1), E(2), E(3), step, ds);
		else
			throw runtime_error;
	}
	else
		throw runtime_error;
}