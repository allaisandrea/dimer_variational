#include "monte_carlo.h"
#include "utilities.h"
#include "states.h"
#include "running_stat.h"
#include <set>

void test_distribution()
{
	unsigned int i, j, io, ie, n_measure = 100000, n_skip = 500, s;
	double Zo, Ze, beta = 5., amp;
	arma::uvec count;
	data_structures<double> ds;
	start_time = time(0x0);
	std::cout << asctime(localtime(&start_time)) << std::endl;
	
	ds.L = 16;
	ds.Nf[0] = 11;
	ds.Nf[1] = 11;
	
	build_graph(ds);

	rng::seed(1);

	homogeneous_state(0., 1., 0., 0., -0.9, beta, false, ds);
	initial_configuration(ds);
	
	rng::seed(1);
	s = 0;
	count.zeros(2 * ds.L * ds.L);
	for(i = 0; i < n_measure; i++)
	{
		for(j = 0; j < n_skip; j++)
		{
			monte_carlo_step(true, true, amp, ds);
		}
		
		for(j = 0; j < ds.Nf[0]; j++)
			count(ds.J[0](j))++;
	}
	count.save("count1.bin");
	ds.w[0].save("w.bin");
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

void test_distribution_1()
{
	unsigned int L = 4, Nu = 0, Nd = 1, c, i, n_measure = 1000000000, n_skip = 1000, which_case, iprint;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05, beta = 1;
	double amp0, amp1, amp2, E0;
	arma::mat Mi[2], X, p;
	arma::vec w;
	data_structures<double> ds;
	arma::uvec buf;
	std::map<arma::uvec, my_pair, classcomp> map;
	std::map<arma::uvec, my_pair, classcomp>::iterator it;
	my_pair *pair;

	ds.L = L;
	build_graph(ds);
	homogeneous_state(dmu, t1, t2, t3, t4, beta, true, ds);
	
	ds.Nf[0] = Nu;
	ds.Nf[1] = Nd;
	initial_configuration(ds);
	
	
	Mi[0] = ds.Mi[0];
	Mi[1] = ds.Mi[1];
	
	E0 = energy(ds);
	iprint = n_measure / 100;
	if(iprint == 0) iprint = 1;
	iprint = 5000;
	for(c = 0; c < n_measure; c++)
	{
		amp0 = phi_amplitude(ds) * abs_squared(arma::det(Mi[0] * ds.M[0]) * arma::det(Mi[1] * ds.M[1]));
		which_case = monte_carlo_step(true, true, amp2, ds);
		amp1 = phi_amplitude(ds) * abs_squared(arma::det(Mi[0] * ds.M[0]) * arma::det(Mi[1] * ds.M[1]));
		if(which_case != 0 && fabs(amp1/amp0/amp2 - 1.) > 1.e-7)
		{
			std::cout << std::setw(4) <<  which_case << std::setw(12) << amp1 / amp0 / amp2 - 1;
// 			test_Mi(ds);
		}
		if(which_case == 0 && fabs(amp1/amp0 - 1.) > 1.e-7)
			std::cout << "err: amp\n";
		
		if(c > 10000)
		{
			amp1 *= exp(E0-energy(ds));
			buf = ds.J[1].rows(0, ds.Nf[1] - 1);
			pair = &map[join_cols(ds.particles, buf)];
			if(pair->value != 0. && fabs(pair->value - amp1) > 1.e-7)
				std::cout << "err:map\n";
			pair->value = amp1;
			pair->count ++;
		}
		if((c + 1) % iprint == 0)
		{
			std::cout << "\r" << std::setw(5) << 100 * (c + 1) / n_measure << " %";
			std::cout.flush();
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
		
	}
	std::cout << "\n";
	
}

// double chop(double x)
// {
// 	return 1.e-10 * round(1.e10 * x);
// }
// void test_gradient_stat()
// {
// 	unsigned int n = 10, nd = 3, i, j;
// 	gradient_running_stat stat(nd);
// 	arma::vec F;
// 	arma::mat Z;
// 	F.randn(n);
// 	Z.randn(n, nd);
// 	for(i = 0; i < n; i++)
// 	{
// 		stat(F(i), trans(Z.row(i)));
// 	}
// 	
// 	std::cout << chop(stat.F - mean(F)) << "\n";
// 	for(i = 0; i < nd; i++)
// 		std::cout << chop(stat.Z(i) - mean(Z.col(i))) << "\n";
// 	
// 	std::cout << chop(stat.FF - accu((F - stat.F) % (F - stat.F)) / n) << "\n";
// 	for(i = 0; i < nd; i++)
// 		std::cout << chop(stat.FZ(i) - accu((F - stat.F) % (Z.col(i) - stat.Z(i))) / n) << "\n";
// 	for(i = 0; i < nd; i++)
// 	for(j = 0; j < nd; j++)
// 		std::cout << chop(stat.ZZ(i, j) - accu((Z.col(i) - stat.Z(i)) % (Z.col(j) - stat.Z(j))) / n) << "\n";
// 	
// 	for(i = 0; i < nd; i++)
// 		std::cout << chop(stat.FFZ(i) - accu((F - stat.F) % (F - stat.F) % (Z.col(i) - stat.Z(i))) / n) << "\n";
// 	
// 	for(i = 0; i < nd; i++)
// 	for(j = 0; j < nd; j++)
// 		std::cout << chop(stat.FZZ(i, j) - accu((F - stat.F) % (Z.col(i) - stat.Z(i)) % (Z.col(j) - stat.Z(j))) / n) << "\n";
// 	
// 	for(i = 0; i < nd; i++)
// 	for(j = 0; j < nd; j++)
// 		std::cout << chop(stat.E(i, j) - accu(((F - stat.F) % (Z.col(i) - stat.Z(i)) - stat.FZ(i)) % ((F - stat.F) % (Z.col(j) - stat.Z(j)) - stat.FZ(j))) / n) << "\n";
// 	
// }


void test_gradient_stat_1()
{
	unsigned int n = 20, i, j, n1 = 1000;
	double F;
	arma::vec x(4), Z(3);
	gradient_running_stat stat(3), stat1(3), stat2(3);
	
	stat1.reset(3);
	
	for(j = 0; j < n1 ; j++)
	{
		stat.reset(3);
		for(i = 0; i< n; i++)
		{
			x = rng::gaussian(4);
			F = x(0) + 2 * x(1) * x(3) - x(0) * x(3) * x(3);
			Z(0) = x(1) * x(2) * x(3) + 2 * x(0) * x(3) - x(1);
			Z(1) = x(0) * x(3) * x(3) + 2 * x(1) * x(3) - x(2);
			Z(2) = x(1) * x(2) * x(1) + 2 * x(0) * x(2) - x(3);
			stat(F, Z);
			stat1(F, Z);
		}
		stat2(1., stat.gradient());
	}
	
	std::cout << stat1.gradient() << "\n";
	std::cout << n1 * stat1.gradient_covariance() << "\n";
// 	std::cout << stat2.ZZ * n / (n - 1) << "\n";
	
}

void test_running_stat()
{
	unsigned int i, j, n = 30, nac = 4;
	running_stat<double> stat(nac);
	arma::vec x;
	x.randn(n);
	for(i = 0; i < n; i++)
		stat(x(i));
	
	for(i = 0; i < nac; i++)
	{
		std::cout << cov(x.rows(0, n - i - 1), x.rows(i, n - 1)) << "\n";
	}
	std::cout << stat.autocorrelation() << "\n";
	
}