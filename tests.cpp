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
	ds.psi.save("psi.bin"); 
	ds.w.save("w.bin");
}

template<class type>
void test_M(const data_structures<type>& ds)
{
	unsigned int i, p;
	arma::uvec e;
	arma::Mat<type> Mu, Md;
	
	Mu.set_size(ds.Nf, ds.Nf);
	Md.set_size(ds.Nf, ds.Nf);
	for(i = 0; i < ds.particles.n_rows; i++)
	{
		p = ds.particles(i) - 2;
		if(p < ds.Nf)
		{
			e << i;
			Mu.row(p) = ds.psi(e, ds.J);
		}
		else if(p < 2 * ds.Nf)
		{
			e << i;
			Md.row(p - ds.Nf) = ds.psi(e, ds.J);
		}
	}
	
	std::cout << "test_M:  ";
	std::cout << std::setw(10) << std::setprecision(3) << norm(Mu - ds.Mu, "fro");
	std::cout << std::setw(10) << std::setprecision(3) << norm(Md - ds.Md, "fro") << "\n";
}

void test_rotate_face_no_step()
{
	// Remember to disable the search for regular configuration.
	unsigned int L = 6, Nf = 3, c;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05;
	arma::umat p;
	data_structures<double> ds;
	
	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	
	c = 0;
	p.set_size(ds.particles.n_rows, 0);
	
	initial_configuration(Nf, ds);
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(30, true, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face( 6, true, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face( 8, true, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face( 7, true, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(20, false, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(10, true, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	std::cout << rotate_face(22, false, false, ds) << "\n";
	p.resize(p.n_rows, p.n_cols + 1);
	p.col(c++) = ds.particles;
	
	p.save("p.bin");
	
	
	for(c = 0; c < 100; c++)
	{
		if(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), false, ds))
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
	
	A.eye(ds.Nf, ds.Nf);
	A -= ds.Mu * ds.Mui;
	std::cout << std::setw(10) << std::setprecision(3) << norm(A, "fro");
	A.eye(ds.Nf, ds.Nf);
	A -= ds.Md * ds.Mdi;
	std::cout << std::setw(10) << std::setprecision(3) << norm(A, "fro") << "\n";
}

void test_rotate_face_with_step()
{
	unsigned int L = 10, Nf = 10, c;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05;
	data_structures<double> ds;
	
	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	
	initial_configuration(Nf, ds);
	test_Mi(ds);
	
	for(c = 0; c < 91; c++)
	{
		while(rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, ds) < 2);
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
	unsigned int L = 4, Nf = 3, c, i, n_measure = 100000000, n_skip = n_measure / 10, which_case;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05;
	double amp0, amp1;
	arma::mat Mui, Mdi, X, p;
	arma::vec w;
	data_structures<double> ds;
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;

	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	X.randn(ds.psi.n_rows, ds.psi.n_cols);
	eig_sym(w, ds.psi, X);
 	ds.phi += 0.05 * rng::gaussian(ds.phi.n_elem);
	
	initial_configuration(Nf, ds);
	
	for(c = 0; c < n_skip; c++)
	{
		rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, ds);
		if(c % 100 == 0)
			std::cout << "\r" << 100. * (c + 1) / n_skip;
	}
	Mui = ds.Mui;
	Mdi = ds.Mdi;
	
	std::cout << "\n";
	
	p.set_size(n_measure, 1);
	for(c = 0; c < n_measure; c++)
	{
	
// 		amp0 = phi_amplitude(ds) * abs_squared(arma::det(Mui * ds.Mu) * arma::det(Mdi * ds.Md));
		which_case = rotate_face(rng::uniform_integer(ds.n_faces), rng::uniform_integer(2), true, ds);
		amp1 = phi_amplitude(ds) * abs_squared(arma::det(Mui * ds.Mu) * arma::det(Mdi * ds.Md));
// 		if(which_case != 0)
// 		{
// 			std::cout << std::setw(12) << amp1 / amp0 << " " << which_case << "\n";
// 			test_Mi(ds);
// 		}
// 		if(which_case == 0 && fabs(amp1/amp0 - 1.) > 1.e-7)
// 			std::cout << "err: amp\n";
		
// 		p(c) =  amp1;
		pair = &map[ds.particles];
		if(pair->value != 0. && fabs(pair->value - amp1) > 1.e-7)
			std::cout << "err:map\n";
		pair->value = amp1;
		pair->count ++;
		
		if(c % 100 == 0)
			std::cout << "\r" << 100. * (c + 1) / n_measure;
	}
	
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