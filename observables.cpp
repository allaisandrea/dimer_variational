#include "data_structures.h"
#include "linear_algebra.h"
#include "utilities.h"


template<class type>
void partition_function_gradient(const data_structures<type> &ds, arma::Col<type> &G)
{
	unsigned int d, nd;
	arma::uvec be, dd;
	arma::Col<type> phi_occ;
	arma::Mat<type> DM;
	double buf;
	
	nd = ds.n_derivatives;
	
	if(	ds.Dphi.n_cols != nd || 
		ds.Dpsi[0].n_slices != nd || 
		ds.Dpsi[1].n_slices != nd || 
		ds.Dw[0].n_cols != nd || 
		ds.Dw[1].n_cols != nd)
		throw std::logic_error("Number of derivatives do not match");
	
	G.zeros(nd);
	
	be = set_to_uvec(ds.boson_edges);
	phi_occ = ds.phi(be);
	for(d = 0; d < nd; d++)
	{
		dd << d;
		G(d) += accu(ds.Dphi(be, dd) / phi_occ);
		
		DM = ds.Dpsi[0].slice(d).submat(ds.fermion_edge[0], ds.J[0]);
		G(d) += trace_of_product(ds.Mi[0], DM);
		
		DM = ds.Dpsi[1].slice(d).submat(ds.fermion_edge[1], ds.J[1]);
		G(d) += trace_of_product(ds.Mi[1], DM);

		G(d) *= 2.;
		
		G(d) -= accu(ds.Dw[0](ds.J[0], dd));
		G(d) -= accu(ds.Dw[1](ds.J[1], dd));
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
	arma::Row<type> V;
	
	arma::uvec e;
	e << destination_f;
	V = ds.psi[s](e, ds.J[s]);
	V -= ds.M[s].row(p);
		
	return conj(ds.phi(destination_b) / ds.phi(origin_b) * (1. + dot(V, ds.Mi[s].col(p))));
	
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
	arma::uvec edges, particles;
	type buf;
	
	buf = 0.;
	for(mu = 0; mu < 2; mu++)
	for(f1 = 0; f1 < ds.n_faces; f1++)
	{
		f2 = ds.adjacent_faces(mu, f1);
		if(mu == 0)
		{
			edges 
				<< ds.face_edges(0, f1)
				<< ds.face_edges(0, f2)
				<< ds.face_edges(1, f2)
				<< ds.face_edges(2, f2)
				<< ds.face_edges(2, f1)
				<< ds.face_edges(3, f1);
		}
		else if(mu == 1)
		{
			edges 
				<< ds.face_edges(0, f1)
				<< ds.face_edges(1, f1)
				<< ds.face_edges(1, f2)
				<< ds.face_edges(2, f2)
				<< ds.face_edges(3, f2)
				<< ds.face_edges(3, f1);
		}
		else
			throw std::logic_error("This should not happen");
		
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
		if(ds.is_boson(particles(2)) && ds.is_fermion(particles(4)))
		{
			ds.to_p_s(particles(4), p, s);
			buf += bf_amplitude(edges(2), edges(4), p, s, edges(4), edges(2), ds);
		}
		if(ds.is_boson(particles(4)) && ds.is_fermion(particles(2)))
		{
			ds.to_p_s(particles(2), p, s);
			buf += bf_amplitude(edges(4), edges(2), p, s, edges(2), edges(4), ds);
		}
		if(ds.is_boson(particles(1)) && ds.is_fermion(particles(5)))
		{
			ds.to_p_s(particles(5), p, s);
			buf += bf_amplitude(edges(1), edges(5), p, s, edges(5), edges(1), ds);
		}
		if(ds.is_boson(particles(5)) && ds.is_fermion(particles(1)))
		{
			ds.to_p_s(particles(1), p, s);
			buf += bf_amplitude(edges(5), edges(1), p, s, edges(1), edges(5), ds);
		}
		if(ds.is_boson(particles(3)) && ds.is_fermion(particles(5)))
		{
			ds.to_p_s(particles(5), p, s);
			buf += bf_amplitude(edges(3), edges(5), p, s, edges(5), edges(3), ds);
		}
		if(ds.is_boson(particles(5)) && ds.is_fermion(particles(3)))
		{
			ds.to_p_s(particles(3), p, s);
			buf += bf_amplitude(edges(5), edges(3), p, s, edges(3), edges(5), ds);
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

