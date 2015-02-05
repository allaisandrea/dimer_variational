#ifndef __rng_h__
#define __rng_h__

#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

class rng
{
public:
	static void seed(unsigned long s)
	{
		gsl_rng_set(r, s);
	}
	
	static unsigned long int uniform_integer(unsigned long int n)
	{
		return gsl_rng_uniform_int(r, n);
	}
	
	static arma::uvec uniform_integer(unsigned int n, unsigned int m)
	{
		unsigned int i;
		arma::uvec v(m);
		for(i = 0; i < v.n_elem; i++)
		{
			v[i] = gsl_rng_uniform_int(r, n);
		}
		return v;
	}
	
	static double uniform()
	{
		return gsl_rng_uniform(r);
	}
	static double gaussian()
	{
		return gsl_ran_ugaussian(r);
	}
	static arma::vec gaussian(unsigned int n)
	{
		unsigned int i;
		arma::vec v(n);
		for(i = 0; i < v.n_elem; i++)
		{
			v[i] = gsl_ran_ugaussian(r);
		}
		return v;
	}
private:
	static gsl_rng *r;
};

#ifdef __main_cpp__
gsl_rng *rng::r = gsl_rng_alloc(gsl_rng_mt19937);
#endif
 
#endif

