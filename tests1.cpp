#include <iostream>
#include <iomanip>
#include <map>
#include <ctime>
#include <set>
#include "data_structures.h"
#include "states.h"
#include "monte_carlo.h"
#include "linear_algebra.h"
#include "utilities.h"
#include "observables.h"
#include "measure_drivers.h"
#include "minimization.h"
#include "rng.h"
#include "running_stat.h"

void test_build_graph()
{
	unsigned int L = 3;
	data_structures<double> ds;
	ds.L = L;
	build_graph(ds);
	std::cout << ds.face_edges << "\n";
	std::cout << ds.adjacent_faces << "\n";
}

void test_homogeneous_state()
{
	unsigned int L = 8;
	double dmu = 0.2, t1 = 0.3, t2 = 1, t3 = 0.2, t4 = 0.9, beta = 1.;
	data_structures<arma::cx_double> ds;
	ds.L = L;
	build_graph(ds);
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
// 	ds.psi[0].save("psi.bin"); 
// 	ds.w[0].save("w.bin");
}

template<class type>
void test_M(const data_structures<type>& ds)
{
	unsigned int i, p;
	arma::uvec e;
	arma::Mat<type> M[2];
	
	M[0].set_size(ds.Nf[0], ds.Nf[0]);
	M[1].set_size(ds.Nf[1], ds.Nf[1]);
	for(i = 0; i < ds.particles.n_rows; i++)
	{
		p = ds.particles(i) - 2;
		if(p < ds.Nf[0])
		{
			e << i;
			M[0].row(p) = ds.psi[0](e, ds.J[0].rows(0, ds.Nf[0] - 1));
		}
		else if(p < ds.Nf[0] + ds.Nf[1])
		{
			e << i;
			M[1].row(p - ds.Nf[0]) = ds.psi[1](e, ds.J[1].rows(0, ds.Nf[1] - 1));
		}
	}
	
	std::cout << "test_M:  ";
	std::cout << std::setw(10) << std::setprecision(3) << norm(M[0] - ds.M[0], "fro");
	std::cout << std::setw(10) << std::setprecision(3) << norm(M[1] - ds.M[1], "fro") << "\n";
}

void test_rotate_face_no_step()
{
	// Remember to disable the search for regular configuration.
	unsigned int L = 6, Nu = 3, Nd = 2, c;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, dummy, beta = 1.;
	arma::umat p;
	data_structures<double> ds;
	
	ds.L = L;
	build_graph(ds);
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
	
	c = 0;
	p.set_size(ds.particles.n_rows, 0);
	
	ds.Nf[0] = Nu;
	ds.Nf[1] = Nd;
	initial_configuration(ds);
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(30, true, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face( 6, true, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face( 8, true, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face( 7, true, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(20, false, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(10, true, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(22, false, false, dummy, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	p.save("p.bin");
	
	
	for(c = 0; c < 100; c++)
	{
		if(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), false, dummy, ds))
			test_M(ds);
	}
}

void test_singular()
{
	arma::mat M;
	M.randn(500, 500);
	std::cout << singular(M) << "\n";
	M.col(0) = M.col(1);
	std::cout << singular(M) << "\n";
	arma::cx_mat M1;
	M1.randn(500, 500);
	std::cout << singular(M1) << "\n";
	M1.col(0) = M1.col(1);
	std::cout << singular(M1) << "\n";
}

template<class type>
void test_Mi(const data_structures<type>& ds)
{
	unsigned int i, p;
	arma::Mat<type> A;
	
	std::cout << "test_Mi: ";
	
	A.eye(ds.Nf[0], ds.Nf[0]);
	A -= ds.M[0] * ds.Mi[0];
	std::cout << std::setw(10) << std::setprecision(3) << norm(A, "fro");
	A.eye(ds.Nf[1], ds.Nf[1]);
	A -= ds.M[1] * ds.Mi[1];
	std::cout << std::setw(10) << std::setprecision(3) << norm(A, "fro") << "\n";
}

template<class type>
void test_edge_assignment(const data_structures<type>& ds)
{
	unsigned int s, e, p;
	std::set<unsigned int> fe[2];
	std::set<unsigned int>::iterator ee;
	
// 	for(ee = ds.boson_edges.begin(); ee != ds.boson_edges.end(); ++ee)
// 	{
// 		if(ds.particles(*ee) != 1)
// 			std::cout << "test_edge_assignment error\n";
// 	}
	
	for(s = 0; s < 2; s++)
	for(p = 0; p < ds.Nf[s]; p++)
	{
		e = ds.fermion_edge[s](p);
		if(fe[s].find(e) != fe[s].end())
			std::cout << "test_edge_assignment error 1\n";
		if(ds.particles(e) != 2 + s * ds.Nf[0] + p)
			std::cout << "test_edge_assignment error 2\n";
		fe[s].insert(e);
	}
	
	for(e = 0; e < ds.particles.n_elem; e++)
	{
		p = ds.particles(e);
		if(p == 0)
		{
// 			if(ds.boson_edges.find(e) != ds.boson_edges.end())
// 				std::cout << "test_edge_assignment error\n";
			if(fe[0].find(e) != fe[0].end())
				std::cout << "test_edge_assignment error 3\n";
			if(fe[1].find(e) != fe[1].end())
				std::cout << "test_edge_assignment error 4\n";
		}
		else if(p == 1)
		{
// 			if(ds.boson_edges.find(e) == ds.boson_edges.end())
// 				std::cout << "test_edge_assignment error\n";
			if(fe[0].find(e) != fe[0].end())
				std::cout << "test_edge_assignment error 5\n";
			if(fe[1].find(e) != fe[1].end())
				std::cout << "test_edge_assignment error 6\n";
		}
		else if(p < 2 + ds.Nf[0] + ds.Nf[1])
		{
			p -= 2;
			s = 0;
			if(p >= ds.Nf[0])
			{
				p-= ds.Nf[0];
				s = 1;
			}
// 			if(ds.boson_edges.find(e) != ds.boson_edges.end())
// 				std::cout << "test_edge_assignment error\n";
			if(ds.fermion_edge[s](p) != e)
				std::cout << "test_edge_assignment error 7\n";
		}
		else
			std::cout << "test_edge_assignment error 8\n";
	}
	
}
	
void test_rotate_face_with_step()
{
	unsigned int L = 10, Nu = 11, Nd = 9, c;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, amp, beta = 1;
	data_structures<double> ds;
	
	ds.L = L;
	build_graph(ds);
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
	
	ds.Nf[0] = Nu;
	ds.Nf[1] = Nd;
	initial_configuration(ds);
	test_Mi(ds);
	
	for(c = 0; c < 100; c++)
	{
		while(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, amp, ds) < 2);
		test_M(ds);
		test_Mi(ds);
		test_edge_assignment(ds);
	}
}

struct classcomp{
	bool operator()(const arma::uvec &a, const arma::uvec &b)const
	{
		unsigned int i, n;
		n = a.n_elem;
		for(i = 0; i < n; i++)
		{
			if(a(i) > b(i))
				return false;
			if(a(i) < b(i))
				return true;
		}
		return false;
	}
};

struct my_pair
{
	my_pair(){count = 0; value = 0.;}
	unsigned int count;
	double value;
};

void test_map()
{
	unsigned int i;
	arma::uvec a;
	std::map<arma::uvec, int, classcomp> map;
	std::map<arma::uvec, int, classcomp>::iterator it;
	
	for(i = 0; i < 1000; i++)
	{
		a = rng::uniform_integer(5, 3);
		map[a] = 1;
	}
	for(it = map.begin(); it != map.end(); it++)
	{
		std::cout << trans(it->first) << "\n";
	}
}

template<class type>
double phi_amplitude(data_structures<type> ds)
{
	unsigned int i;
	type Phi = 1;
	for(i = 0; i < ds.particles.n_elem; i++)
	{
		if(ds.particles(i) == 1)
			Phi *= ds.phi(i);
	}
	return abs_squared(Phi);
}

void test_correct_distribution()
{
	unsigned int L = 4, Nu = 2, Nd = 2, c, i, n_measure = 1<<24, n_skip = n_measure / 16, which_case;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, beta = 1;
	double amp0, amp1, amp2;
	arma::mat Mi[2], X, p;
	arma::vec w;
	data_structures<double> ds;
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;

	ds.L = L;
	build_graph(ds);
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
	X.randn(ds.psi[0].n_rows, ds.psi[0].n_cols);
	eig_sym(w, ds.psi[0], X);
	ds.psi[1] = ds.psi[0];
 	ds.phi += 0.05 * rng::gaussian(ds.phi.n_elem);
	
	ds.Nf[0] = Nu;
	ds.Nf[1] = Nd;
	initial_configuration(ds);
	
	for(c = 0; c < n_skip; c++)
	{
		rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, amp2, ds);
		if((c + 1) % (n_skip / 128) == 0)
		{
			std::cout << "\r" << std::setw(5) << 100 * (c + 1) / n_skip << " %";
			std::cout.flush();
		}
	}
	std::cout << "\n";
	
	Mi[0] = ds.Mi[0];
	Mi[1] = ds.Mi[1];
	
	
	for(c = 0; c < n_measure; c++)
	{
	
		amp0 = phi_amplitude(ds) * abs_squared(arma::det(Mi[0] * ds.M[0]) * arma::det(Mi[1] * ds.M[1]));
		which_case = rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, amp2, ds);
		amp1 = phi_amplitude(ds) * abs_squared(arma::det(Mi[0] * ds.M[0]) * arma::det(Mi[1] * ds.M[1]));
		if(which_case != 0 && fabs(amp1/amp0/amp2 - 1.) > 1.e-7)
		{
			std::cout << std::setw(4) <<  which_case << std::setw(12) << amp1 / amp0 / amp2 - 1;
			test_Mi(ds);
		}
		if(which_case == 0 && fabs(amp1/amp0 - 1.) > 1.e-7)
			std::cout << "err: amp\n";
		
		pair = &map[ds.particles];
		if(pair->value != 0. && fabs(pair->value - amp1) > 1.e-7)
			std::cout << "err:map\n";
		pair->value = amp1;
		pair->count ++;
		
		if((c + 1) % (n_measure / 128) == 0)
		{
			std::cout << "\r" << std::setw(5) << 100 * (c + 1) / n_measure << " %";
			std::cout.flush();
		}
		
	}
	std::cout << "\n";
	
	p.set_size(2, map.size());
	c = 0;
	for(it = map.begin(); it != map.end(); it++)
	{
		p(0, c) = (*it).second.value;
		p(1, c) = (*it).second.count;
		c++;
	}
	p.save("tally.bin");
}

void test_apriori_swap_proposal()
{
	unsigned int n = 5, no = 3, i, n_measure = 100000, io, ie, s = 1;
	arma::mat p;
	double max_w, Zo, Ze;
	
	data_structures<double> ds;
	
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;
	
	ds.Nf[s] = no;
	ds.w[s].randn(n);
	ds.J[s] = sort_index(ds.w[s]);
	compute_state_weights(ds);
	
	max_w = max(ds.w[s]);
	
	for(i = s; i < n_measure; i++)
	{
		if(apriori_swap_proposal(s, ds, io, Zo, ie, Ze))
		{
			swap(ds.J[s](io), ds.J[s](ie));
			swap(ds.Epw[s](io), ds.Epw[s](ie));
			swap(ds.Emw[s](io), ds.Emw[s](ie));
			ds.Zo[s] = Zo;
			ds.Ze[s] = Ze;
		}
		pair = &map[ds.J[s]];
		pair->value = exp(max_w - accu(ds.w[s](ds.J[s].rows(0, ds.Nf[s] - 1))));
		pair->count ++;
	}
	
	p.set_size(2, map.size());
	i = 0;
	for(it = map.begin(); it != map.end(); it++)
	{
		p(0, i) = (*it).second.value;
		p(1, i) = (*it).second.count;
		i++;
	}
	p.save("tally.bin");
	
}

void my_print(const arma::uvec &v)
{
	unsigned int i;
	for(i = 0; i < v.n_elem; i++)
		std::cout << std::setw(3) << v(i);
	std::cout << "\n";
}

template<class type>
double energy(data_structures<type> &ds)
{
	double E;
	unsigned int i, s;
	E = 0;
	for(s = 0; s < 2; s++)
	for(i = 0; i < ds.Nf[s]; i++)
		E += ds.w[s](ds.J[s](i));
	return E;
}

void test_swap_states()
{
	unsigned int L = 4, Nu = 2, Nd = 2, c, i, n_measure = 1<<20, n_skip = n_measure / 16, which_case, s;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, beta = 1;
	double amp0, amp1, amp2, E0;
	arma::mat Mi[2], X, p;
	arma::vec w;
	arma::uvec J0[2], J1[2];
	data_structures<double> ds;
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;

	ds.L = L;
	build_graph(ds);
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
	X.randn(ds.psi[0].n_rows, ds.psi[0].n_cols);
	eig_sym(w, ds.psi[0], X);
	ds.psi[1] = ds.psi[0];
 	ds.phi += 0.05 * rng::gaussian(ds.phi.n_elem);
	
	ds.Nf[0] = Nu;
	ds.Nf[1] = Nd;
	initial_configuration(ds); 
	
	for(c = 0; c < n_skip; c++)
	{
		swap_states(rng::uniform_integer(2), amp2, ds);
// 		if((c + 1) % (n_skip / 128) == 0)
		{
			std::cout << "\r" << std::setw(5) << 100 * (c + 1) / n_skip << " %";
			std::cout.flush();
		}
	}
	std::cout << "\n";
	
	Mi[0] = ds.Mi[0];
	Mi[1] = ds.Mi[1];
	E0 = energy(ds);
	classcomp my_less;
	for(c = 0; c < n_measure; c++)
	{
		s = rng::uniform_integer(2);
		J0[0] = ds.J[0];
		J0[1] = ds.J[1];
		amp0 = phi_amplitude(ds) * abs_squared(arma::det(Mi[0] * ds.M[0]) * arma::det(Mi[1] * ds.M[1]));
		which_case = swap_states(s, amp2, ds);
		J1[0] = ds.J[0];
		J1[1] = ds.J[1];
		amp1 = phi_amplitude(ds) * abs_squared(arma::det(Mi[0] * ds.M[0]) * arma::det(Mi[1] * ds.M[1]));
		if(which_case != 0 && fabs(amp1/amp0/amp2 - 1.) > 1.e-7)
		{
			test_M(ds);
			test_Mi(ds);
			std::cout << s << std::setw(12) << amp1 / amp0 / amp2 - 1. << "\n";
			my_print(J0[s]);
			my_print(J1[s]);
			std::cout << "\n";
		}
		
		if(which_case == 0 && fabs(amp1/amp0 - 1.) > 1.e-7)
			std::cout << "err: amp\n";
		
		pair = &map[arma::join_vert(ds.J[0].rows(0, ds.Nf[0] - 1), ds.J[1].rows(0, ds.Nf[1] - 1))];
		amp1 *= exp(E0-energy(ds));
		if(pair->value != 0. && fabs(pair->value - amp1) > 1.e-7)
			std::cout << "err:map\n";
		pair->value = amp1;
		pair->count ++;
		
		if((c + 1) % (n_measure / 128) == 0)
		{
			std::cout << "\r" << std::setw(5) << 100 * (c + 1) / n_measure << " %";
			std::cout.flush();
		}
		
	}
	std::cout << "\n";
	
	p.set_size(2, map.size());
	c = 0;
	for(it = map.begin(); it != map.end(); it++)
	{
		p(0, c) = (*it).second.value;
		p(1, c) = (*it).second.count;
		c++;
	}
	p.save("tally.bin");
}


void test_eigensystem_variation()
{
	unsigned int n = 50;
	double eps = 1.e-7;
	arma::mat H, U, U1, U2, V, dU1, dU2;
	arma::vec w, w1, w2, dw1, dw2;
	H.randn(n, n);
	H += trans(H);
	V.randn(n, n);
	V += trans(V);
	
	eig_sym(w, U, H);
	
	eigensystem_variation(U, w, V, dU1, dw1);
	
	eig_sym(w1, U1, H + eps * V);
	eig_sym(w2, U2, H - eps * V);
	
	dw2 = (w1 - w2) / 2. / eps;
	dU2 = (U1 - U2) / 2. / eps;
	
	std::cout << norm(dw1 - dw2, "fro") << "\n";
	std::cout << norm(dU1 - dU2, "fro") << "\n";
}

void test_homogeneous_state_derivatives()
{
	unsigned int i, j, s;
	double eps = 1.e-7, beta = 1;
	arma::vec tt;
	arma::mat dt;
	arma::mat psi[2], Dpsi[2], Dpsi_approx[2];
	arma::vec Dw[2], Dw_approx[2], overlaps;
	data_structures<double> ds;
	
	ds.L = 4;
	
	dt = arma::eye<arma::mat>(5, 5);
	
	for(i = 0; i < 5; i++)
	{
		tt << 0. << 1. << 0.2 << 0.0 << 1.0;
		
		homogeneous_state(tt(0), tt(1), tt(2), tt(3), tt(4), beta, true, ds);
		for(s = 0; s < 2; s++)
		{
			psi[s] = ds.psi[s];
			Dpsi[s] = ds.Dpsi[s].slice(i);
			Dw[s] = ds.Dw[s].col(i);
		}
		
		
		tt += eps * dt.col(i);
		homogeneous_state(tt(0), tt(1), tt(2), tt(3), tt(4), beta, true, ds);
		for(s = 0; s< 2; s++)
		{
			overlaps.set_size(ds.psi[s].n_cols);
			for(j = 0; j < ds.psi[s].n_cols; j++)
			{
				overlaps(j) = dot(psi[s].col(j), ds.psi[s].col(j));
				if(fabs(fabs(overlaps(j)) - 1.) > 1.e-7)
					std::cout << "overlap 1: " << std::setw(5) << overlaps(j) << std::setw(5) << fabs(overlaps(j)) - 1. << "\n";
				overlaps(j) /= fabs(overlaps(j));
			}
			Dpsi_approx[s] = ds.psi[s] * arma::diagmat(overlaps);
			Dw_approx[s] = ds.w[s];
		}
		
		
		tt -= 2. * eps * dt.col(i);
		homogeneous_state(tt(0), tt(1), tt(2), tt(3), tt(4), beta, true, ds);
		for(s = 0; s< 2; s++)
		{
			overlaps.set_size(ds.psi[s].n_cols);
			for(j = 0; j < ds.psi[s].n_cols; j++)
			{
				overlaps(j) = dot(psi[s].col(j), ds.psi[s].col(j));
				if(fabs(fabs(overlaps(j)) - 1.) > 1.e-7)
					std::cout << "overlap 1: " << std::setw(5) << overlaps(j) << std::setw(5) << fabs(overlaps(j)) - 1. << "\n";
				overlaps(j) /= fabs(overlaps(j));
			}
			Dpsi_approx[s] -= ds.psi[s] * arma::diagmat(overlaps);
			Dw_approx[s] -= ds.w[s];
		}
		
		tt += eps * dt.col(i);
		
		Dpsi_approx[0] /= 2 * eps;
		Dpsi_approx[1] /= 2 * eps;
		Dw_approx[0] /= 2 * eps;
		Dw_approx[1] /= 2 * eps;
		
		std::cout << norm(Dw[0] - Dw_approx[0], "fro") << "\n";
		std::cout << norm(Dw[1] - Dw_approx[1], "fro") << "\n";
		std::cout << norm(Dpsi_approx[0] - Dpsi[0], "fro") << "\n";
		std::cout << norm(Dpsi_approx[1] - Dpsi[1], "fro") << "\n";
		
// 		for(j = 0; j < Dpsi[0].n_cols; j++)
// 		{
// 			std::cout << norm(Dpsi_approx[0].col(j) - Dpsi[0].col(j), "fro") << "\n";
// 			std::cout << join_rows(Dpsi_approx[0].col(j), Dpsi[0].col(j)) <<  "\n";
// 		}
		
		std::cout << "\n";
	}
}


void test_monte_carlo_driver()
{
	unsigned int i, j, n_measure = 100000, n_skip = 1000, n_points = 20, n_observables;
	double dmu = 0, t1 = 1., t2 = 0, t3 = 0, t4 = 0.5, beta = 10.;
	double E, sE;
	arma::mat F, dZ;
	arma::umat J[2];
	arma::cube sF;
	arma::cube G, sG, EE;
	data_structures<double> ds;
	observables_vector_real observables;
	arma::vec coefficients, buf1, buf2;
	std::string suffix;
	
	// Doping: 0.1 holes / site => ds.Nf[s] = 0.05 * ds.L * ds.L
	ds.L = 24;
	ds.Nf[0] = 29;
	ds.Nf[1] = 29;
	
	observables.push_back(&boson_hopping);
// 	observables.push_back(&boson_potential);
	observables.push_back(&fermion_hopping_1);
	observables.push_back(&fermion_hopping_2);
	observables.push_back(&fermion_hopping_3);
	n_observables = observables.size();
	
	build_graph(ds);

	rng::seed(1);
	G.zeros(5, n_points, n_observables);
	sG.zeros(5, n_points, n_observables);
	EE.zeros(2, n_points, n_observables); 
	sF.zeros(20, n_observables, n_points);
	for(i = 0; i < n_points; i++)
	{
		std::cout << "point " << i + 1 << "..." << std::endl;
		t2 = -0.2 +  0.4 * i / (n_points - 1.);
		homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
		monte_carlo_driver(n_measure, n_skip, true, false, observables, ds, F, dZ, J);
		
		autocorrelations(F, sF.slice(i));
		for(j = 0; j < n_observables; j++)
		{
			coefficients.zeros(n_observables);
			coefficients(j) = 1.;
			total_energy(F, dZ, coefficients, E, sE, buf1, buf2);
			G.slice(j).col(i) = buf1;
			sG.slice(j).col(i) = buf2;
			EE(0, i, j) = E;
			EE(1, i, j) = sE;
		}
		std::cout << "done\n";
		suffix = "_07";
		
		sF.save("sF"+suffix+".bin");
		G.save("G"+suffix+".bin");
		sG.save("sG"+suffix+".bin");
		EE.save("E"+suffix+".bin");
	}
}


void test_states_autocorrelation()
{
	unsigned int i, j, n_measure = 10000, n_skip = 10000, n_points = 20, n_observables, start_time;
	double dmu = 0, t1 = 1., t2 = 0.01, t3 = 0, t4 = 1., beta = 50.;
	double E, sE;
	arma::mat F, dZ;
	arma::umat J[2];
	data_structures<double> ds;
	observables_vector_real observables;
	arma::vec coefficients, buf1, buf2;
	
	
	
	ds.L = 24;
	ds.Nf[0] = 29;
	ds.Nf[1] = 29;
	
	observables.push_back(&boson_hopping);
// 	observables.push_back(&boson_potential);
	observables.push_back(&fermion_hopping_1);
	observables.push_back(&fermion_hopping_2);
	observables.push_back(&fermion_hopping_3);
	n_observables = observables.size();
	
	build_graph(ds);

	rng::seed(1);

	
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
	start_time = std::clock();
	monte_carlo_driver(n_measure, n_skip, true, true, observables, ds, F, dZ, J);
	std::cout << 1. * (std::clock() - start_time) / CLOCKS_PER_SEC << "\n";
	J[0].save("Ju.bin");
	J[1].save("Jd.bin");
	ds.w[0].save("w.bin");
}

extern "C"{
	void dger_(int*, int*, double*, double*, int *, double*, int*, double*, int*);
}

void test_rank_1_update()
{
	arma::mat M;
	arma::rowvec v;
	arma::colvec u;
	
	int i, start_time, m = 500000, n = 20, inc = 1;
	double alpha = 1.5;
	M.randn(n, n);
	v.randn(n);
	u.randn(n);
	start_time = std:: clock();
	for(i = 0; i < m; i++)
	{
		M * u;
	}
	std::cout << 1. * (std::clock() - start_time) / CLOCKS_PER_SEC << "\n";
	
	start_time = std:: clock();
	for(i = 0; i < m; i++)
	{
		dger_(&n, &n, &alpha, v.memptr(), &inc, u.memptr(), &inc, M.memptr(), &n);
	}
	std::cout << 1. * (std::clock() - start_time) / CLOCKS_PER_SEC << "\n";
	
	start_time = std:: clock();
	for(i = 0; i < m; i++)
	{
		rank_1_update(alpha, u, v, M);
	}
	std::cout << 1. * (std::clock() - start_time) / CLOCKS_PER_SEC << "\n";
}

void f(const arma::vec& x, running_stat<double> & y, double &p)
{
	unsigned int i;
	double y0;
	y0 = tanh(x(0) * x(0) + 0.2 * x(1) * x(1) + 0.5 * x(0) * x(1));
	for(i = 0; i < 20; i++)
	{
		y(y0 + 0.1 * rng::gaussian());
	}
}

void test_minimization()
{
	double x;
	arma::mat p(2, 3);
	p.randn();
	simplex_minimize<double>(p, f, x);
}

void test_running_stat()
{
	unsigned int i, j;
	running_stat<double> rs1, rs2;
	arma::vec x1, x2, x3;
	
	x1.randn(10);
	x2.randn(15);
	
	for(i = 0; i < x1.n_rows; i++)
		rs1(x1(i));
	
	for(i = 0; i < x2.n_rows; i++)
		rs2(x2(i));
	
	std::cout << mean(x1) << "\n";
	std::cout << rs1.mean() << "\n";
	std::cout << var(x1, 1) << "\n";
	std::cout << rs1.second_moment() << "\n";
	
	rs1(rs2);
	std::cout << mean(join_cols(x1, x2)) << "\n";
	std::cout << rs1.mean() << "\n";
	std::cout << var(join_cols(x1, x2), 1) << "\n";
	std::cout << rs1.second_moment() << "\n";
	
	rs2.reset();
	x3.randn(7);
	for(i = 0; i < x3.n_rows; i++)
		rs2(x3(i));
	
	rs1(rs2);
	std::cout << mean(join_cols(join_cols(x1, x2), x3)) << "\n";
	std::cout << rs1.mean() << "\n";
	std::cout << var(join_cols(join_cols(x1, x2), x3), 1) << "\n";
	std::cout << rs1.second_moment() << "\n";
}

struct test_params
{
	unsigned int n_measure;
	unsigned int n_dim;
};

void h_parallel(const arma::vec& x, running_stat<double> & y, test_params &p)
{
	unsigned int i;
	double y0;
	y0 = tanh(x(0) * x(0) + 0.2 * x(1) * x(1) + 0.5 * x(0) * x(1));
	for(i = 0; i < p.n_measure; i++)
	{
		y(y0 + 0.1 * rng::gaussian());
	}
}

void f_parallel(const arma::vec& x, running_stat<double> & y, test_params &p)
{
	running_stat<double> yy[1];
	arma::vec xx;
	xx = x;
	MPI_Send(xx.memptr(), xx.n_elem * sizeof(double), MPI_BYTE, 1, 1, MPI_COMM_WORLD);
	h_parallel(x, y, p);
	MPI_Recv(&yy[0], sizeof(yy[0]), MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	y(yy[0]);
}


void test_minimization_parallel()
{
	int rank;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(rank == 0)
	{
		test_params p;
		arma::mat x0(2, 3);
		p.n_measure = 20;
		p.n_dim = 2;
		
		MPI_Send(&p, sizeof(p), MPI_BYTE, 1, 1, MPI_COMM_WORLD);
		
		x0.randn();
		simplex_minimize<test_params>(x0, f_parallel, p);
	}
	else
	{
		test_params p;
		arma::vec x;
		running_stat<double> y;
		MPI_Recv(&p, sizeof(p), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		x.set_size(p.n_dim);
		
		while(true)
		{
			MPI_Recv(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			y.reset();
			h_parallel(x, y, p);
			
			MPI_Send(&y, sizeof(y), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
		}
		
	}
}
