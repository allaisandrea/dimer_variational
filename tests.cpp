#include <iostream>
#include <iomanip>
#include <map>
#include "data_structures.h"
#include "states.h"
#include "monte_carlo.h"
#include "linear_algebra.h"
#include "utilities.h"

void test_build_graph()
{
	unsigned int L = 3;
	data_structures<double> ds;
	build_graph(L, ds);
	std::cout << ds.face_edges << "\n";
	std::cout << ds.adjacent_faces << "\n";
}

void test_homogeneous_state()
{
	unsigned int L = 10;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05;
	data_structures<arma::cx_double> ds;
	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	ds.psi[0].save("psi.bin"); 
	ds.w[0].save("w.bin");
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
			M[0].row(p) = ds.psi[0](e, ds.J[0]);
		}
		else if(p < ds.Nf[0] + ds.Nf[1])
		{
			e << i;
			M[1].row(p - ds.Nf[0]) = ds.psi[1](e, ds.J[1]);
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
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, dummy;
	arma::umat p;
	data_structures<double> ds;
	
	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	
	c = 0;
	p.set_size(ds.particles.n_rows, 0);
	
	initial_configuration(Nu, Nd, ds);
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

void test_rotate_face_with_step()
{
	unsigned int L = 10, Nu = 10, Nd = 10, c;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, amp;
	data_structures<double> ds;
	
	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	
	initial_configuration(Nu, Nd, ds);
	test_Mi(ds);
	
	for(c = 0; c < 91; c++)
	{
		while(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, amp, ds) < 2);
		test_M(ds);
		test_Mi(ds);
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
	unsigned int L = 4, Nu = 3, Nd = 2, c, i, n_measure = 1<<25, n_skip = n_measure / 16, which_case;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05;
	double amp0, amp1, amp2;
	arma::mat Mi[2], X, p;
	arma::vec w;
	data_structures<double> ds;
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;

	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	X.randn(ds.psi[0].n_rows, ds.psi[0].n_cols);
	eig_sym(w, ds.psi[0], X);
	ds.psi[1] = ds.psi[0];
 	ds.phi += 0.05 * rng::gaussian(ds.phi.n_elem);
	
	initial_configuration(Nu, Nd, ds);
	
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
	
	
	
	p.set_size(n_measure, 1);
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
	
	p.save("p.bin");
	
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
	unsigned int n = 5, no = 3, i, n_measure = 100000, io, ie;
	double max_w;
	arma::vec w;
	arma::uvec Jo, Je;
	arma::mat p;
	
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;
	
	w.randn(n);
	max_w = max(w);
	Jo = arma::linspace<arma::uvec>(0, no - 1, no);
	Je = arma::linspace<arma::uvec>(no, n - 1, n - no);
	
	for(i = 0; i < n_measure; i++)
	{
		if(apriori_swap_proposal(w, Jo, Je, io, ie))
		{
			swap(Jo(io), Je(ie));
		}
		pair = &map[Jo];
		pair->value = exp(max_w - accu(w(Jo)));
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