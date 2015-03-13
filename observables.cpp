#include "data_structures.h"
#include "linear_algebra.h"
#include "utilities.h"


template<class type>
void partition_function_gradient(const data_structures<type> &ds, arma::Col<type> &G)
{
	unsigned int d, nd, i, s;
	const unsigned int *J;
	static arma::Mat<type> DM[2];
	type buf;
	
	nd = ds.n_derivatives;
	
	if(	ds.Dphi.n_cols != nd || 
		ds.Dpsi[0].n_slices != nd || 
		ds.Dpsi[1].n_slices != nd || 
		ds.Dw[0].n_cols != nd || 
		ds.Dw[1].n_cols != nd)
		throw std::logic_error("Number of derivatives do not match");
	
	G.set_size(nd);
	DM[0].set_size(ds.Nf[0], ds.Nf[0]);
	DM[1].set_size(ds.Nf[1], ds.Nf[1]);
	
	for(d = 0; d < nd; d++)
	{
		buf = 0.;
		for(i = 0; i < ds.particles.n_elem; i++)
		{
			if(ds.particles(i) == 1)
				buf += ds.Dphi(i, d) / ds.phi(i);
		}
		
		for(s = 0; s < 2; s++)
		{
			J = ds.J[s].memptr();
			copy_matrix_sparse(ds.Nf[s], ds.Nf[s], ds.fermion_edge[s].memptr(), J, ds.Dpsi[s].memptr() + d * ds.Dpsi[s].n_rows * ds.Dpsi[s].n_cols, ds.Dpsi[s].n_rows, DM[s].memptr(), DM[s].n_rows);
			buf += 2. * trace_of_product(ds.Mi[s], DM[s]);
			for(i = 0; i < ds.Nf[s]; i++)
			{
				buf -= ds.Dw[s](J[i], d);
			}
		}		
		G(d) = buf;
	}
	
}

template<class type>
type bb_amplitude(
	unsigned int origin1, 
	unsigned int destination1, 
	unsigned int origin2, 
	unsigned int destination2, 
	const data_structures<type> &ds)
{
	return conj(ds.phi(destination1) * ds.phi(destination2) / ds.phi(origin1) / ds.phi(origin2));
}

template<class type>
type bf_amplitude(
	unsigned int origin_b, 
	unsigned int destination_b, 
	unsigned int p,
	unsigned int s,
	unsigned int origin_f, 
	unsigned int destination_f, 
	const data_structures<type> &ds)
{
	static arma::Row<type> V[2];
	static arma::Col<type> U[2];
	
	arma::uvec e;
	e << destination_f;
	
	V[s].set_size(ds.M[s].n_cols);
	copy_vector_sparse(ds.M[s].n_cols, ds.J[s].memptr(), ds.psi[s].memptr() + destination_f, ds.psi[s].n_rows, V[s].memptr(), V[s].n_rows);
	add_to_vector(ds.M[s].n_cols, (type) -1., ds.M[s].memptr() + p, ds.M[s].n_rows, V[s].memptr(), V[s].n_rows);
	
	U[s].set_size(ds.M[s].n_rows);
	copy_vector(ds.M[s].n_rows, ds.Mi[s].colptr(p), 1, U[s].memptr(), 1);
	return conj(ds.phi(destination_b) / ds.phi(origin_b) * (1. + dot(V[s], U[s])));
	
}

template<class type>
type boson_hopping(const data_structures<type> &ds)
{
	unsigned int f;
	arma::uvec edges, particles;
	type buf;
	
	buf = 0.;
	for(f = 0; f < ds.n_faces; f++)
	{
		edges = ds.face_edges.col(f);
		particles = ds.particles(edges);
		
		if(     ds.is_boson(particles(0)) && ds.is_boson(particles(2)))
			buf += bb_amplitude(edges(0), edges(1), edges(2), edges(3), ds);
		if(ds.is_boson(particles(1)) && ds.is_boson(particles(3)))
			buf += bb_amplitude(edges(1), edges(2), edges(3), edges(0), ds);
	}
	
	return buf / (double) ds.n_faces;
}


template<class type>
type boson_potential(const data_structures<type> &ds)
{
	unsigned int f;
	arma::uvec particles;
	type buf;
	
	buf = 0.;
	for(f = 0; f < ds.n_faces; f++)
	{
		particles = ds.particles(ds.face_edges.col(f));
		if((ds.is_boson(particles(0)) && ds.is_boson(particles(2))) || (ds.is_boson(particles(1)) && ds.is_boson(particles(3))))
			buf += 1.;
		else
			buf += 0.;
	}
	
	return buf / (double) ds.n_faces;
}


template<class type>
type fermion_hopping_1(const data_structures<type> &ds)
{
	unsigned int p, s, f;
	arma::uvec edges, particles;
	type buf;
	
	buf = 0.;
	for(f = 0; f < ds.n_faces; f++)
	{
		edges = ds.face_edges.col(f);
		particles = ds.particles(edges);
		
		
		if(ds.is_boson(particles(0)) && ds.is_fermion(particles(2)))
		{
			ds.to_p_s(particles(2), p, s);
			buf += bf_amplitude(edges(0), edges(2), p, s, edges(2), edges(0), ds);
		}
		if(ds.is_boson(particles(2)) && ds.is_fermion(particles(0)))
		{
			ds.to_p_s(particles(0), p, s);
			buf += bf_amplitude(edges(2), edges(0), p, s, edges(0), edges(2), ds);
		}
		if(ds.is_boson(particles(1)) && ds.is_fermion(particles(3)))
		{
			ds.to_p_s(particles(3), p, s);
			buf += bf_amplitude(edges(1), edges(3), p, s, edges(3), edges(1), ds);
		}
		if(ds.is_boson(particles(3)) && ds.is_fermion(particles(1)))
		{
			ds.to_p_s(particles(1), p, s);
			buf += bf_amplitude(edges(3), edges(1), p, s, edges(1), edges(3), ds);
		}
	}
	
	return buf / (double) ds.n_faces;
}

template<class type>
type fermion_hopping_2(const data_structures<type> &ds)
{
	unsigned int p, s, f;
	arma::uvec edges, particles;
	type buf;
	
	buf = 0.;
	for(f = 0; f < ds.n_faces; f++)
	{
		edges = ds.face_edges.col(f);
		particles = ds.particles(edges);
		
		
		if(ds.is_boson(particles(0)) && ds.is_fermion(particles(2)))
		{
			ds.to_p_s(particles(2), p, s);
			buf += bf_amplitude(edges(0), edges(1), p, s, edges(2), edges(3), ds);
			buf += bf_amplitude(edges(0), edges(3), p, s, edges(2), edges(1), ds);
		}
		if(ds.is_boson(particles(2)) && ds.is_fermion(particles(0)))
		{
			ds.to_p_s(particles(0), p, s);
			buf += bf_amplitude(edges(2), edges(1), p, s, edges(0), edges(3), ds);
			buf += bf_amplitude(edges(2), edges(3), p, s, edges(0), edges(1), ds);
		}
		if(ds.is_boson(particles(1)) && ds.is_fermion(particles(3)))
		{
			ds.to_p_s(particles(3), p, s);
			buf += bf_amplitude(edges(1), edges(2), p, s, edges(3), edges(0), ds);
			buf += bf_amplitude(edges(1), edges(0), p, s, edges(3), edges(2), ds);
		}
		if(ds.is_boson(particles(3)) && ds.is_fermion(particles(1)))
		{
			ds.to_p_s(particles(1), p, s);
			buf += bf_amplitude(edges(3), edges(2), p, s, edges(1), edges(0), ds);
			buf += bf_amplitude(edges(3), edges(0), p, s, edges(1), edges(2), ds);
		}
	}
	return buf / (double) ds.n_faces;
}

template<class type>
type fermion_hopping_3(const data_structures<type> &ds)
{
	unsigned int f1, f2, mu,  p, s;
	unsigned int edges[6], particles[6];
	type buf;
	
	buf = 0.;
	for(mu = 0; mu < 2; mu++)
	for(f1 = 0; f1 < ds.n_faces; f1++)
	{
		f2 = ds.adjacent_faces(mu, f1);
		if(mu == 0)
		{
			edges[0] =  ds.face_edges(0, f1);
			edges[1] =  ds.face_edges(0, f2);
			edges[2] =  ds.face_edges(1, f2);
			edges[3] =  ds.face_edges(2, f2);
			edges[4] =  ds.face_edges(2, f1);
			edges[5] =  ds.face_edges(3, f1);
		}
		else if(mu == 1)
		{
			edges[0] =  ds.face_edges(0, f1);
			edges[1] =  ds.face_edges(1, f1);
			edges[2] =  ds.face_edges(1, f2);
			edges[3] =  ds.face_edges(2, f2);
			edges[4] =  ds.face_edges(3, f2);
			edges[5] =  ds.face_edges(3, f1);
		}
		else
			throw std::logic_error("This should not happen");
		particles[0] = ds.particles(edges[0]);
		particles[1] = ds.particles(edges[1]);
		particles[2] = ds.particles(edges[2]);
		particles[3] = ds.particles(edges[3]);
		particles[4] = ds.particles(edges[4]);
		particles[5] = ds.particles(edges[5]);
		
		if(ds.is_boson(particles[0]) && ds.is_fermion(particles[2]))
		{
			ds.to_p_s(particles[2], p, s);
			buf += bf_amplitude(edges[0], edges[2], p, s, edges[2], edges[0], ds);
		}
		if(ds.is_boson(particles[2]) && ds.is_fermion(particles[0]))
		{
			ds.to_p_s(particles[0], p, s);
			buf += bf_amplitude(edges[2], edges[0], p, s, edges[0], edges[2], ds);
		}
		if(ds.is_boson(particles[2]) && ds.is_fermion(particles[4]))
		{
			ds.to_p_s(particles[4], p, s);
			buf += bf_amplitude(edges[2], edges[4], p, s, edges[4], edges[2], ds);
		}
		if(ds.is_boson(particles[4]) && ds.is_fermion(particles[2]))
		{
			ds.to_p_s(particles[2], p, s);
			buf += bf_amplitude(edges[4], edges[2], p, s, edges[2], edges[4], ds);
		}
		if(ds.is_boson(particles[1]) && ds.is_fermion(particles[5]))
		{
			ds.to_p_s(particles[5], p, s);
			buf += bf_amplitude(edges[1], edges[5], p, s, edges[5], edges[1], ds);
		}
		if(ds.is_boson(particles[5]) && ds.is_fermion(particles[1]))
		{
			ds.to_p_s(particles[1], p, s);
			buf += bf_amplitude(edges[5], edges[1], p, s, edges[1], edges[5], ds);
		}
		if(ds.is_boson(particles[3]) && ds.is_fermion(particles[5]))
		{
			ds.to_p_s(particles[5], p, s);
			buf += bf_amplitude(edges[3], edges[5], p, s, edges[5], edges[3], ds);
		}
		if(ds.is_boson(particles[5]) && ds.is_fermion(particles[3]))
		{
			ds.to_p_s(particles[3], p, s);
			buf += bf_amplitude(edges[5], edges[3], p, s, edges[3], edges[5], ds);
		}
	}
	
	return buf / (double) ds.n_faces;
}


template
void partition_function_gradient<double>(const data_structures<double> &ds, arma::Col<double> &G);

template
void partition_function_gradient<arma::cx_double>(const data_structures<arma::cx_double> &ds, arma::Col<arma::cx_double> &G);

template
double boson_hopping<double>(const data_structures<double> &ds);

template
arma::cx_double boson_hopping<arma::cx_double>(const data_structures<arma::cx_double> &ds);

template
double boson_potential<double>(const data_structures<double> &ds);

template
arma::cx_double boson_potential<arma::cx_double>(const data_structures<arma::cx_double> &ds);

template
double fermion_hopping_1<double>(const data_structures<double> &ds);

template
arma::cx_double fermion_hopping_1<arma::cx_double>(const data_structures<arma::cx_double> &ds);

template
double fermion_hopping_2<double>(const data_structures<double> &ds);

template
arma::cx_double fermion_hopping_2<arma::cx_double>(const data_structures<arma::cx_double> &ds);

template
double fermion_hopping_3<double>(const data_structures<double> &ds);

template
arma::cx_double fermion_hopping_3<arma::cx_double>(const data_structures<arma::cx_double> &ds);

