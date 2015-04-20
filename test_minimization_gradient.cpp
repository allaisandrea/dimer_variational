#include "minimization_gradient.h"
#include "rng.h"
struct parameters_t
{
int i;
};

void f(const arma::vec& x, running_stat<double> & y, parameters_t &p)
{
	unsigned int i;
	double y0, s = 1e-2;
	y0 = 1 + sin(x(0)) * sin(x(1)) / sqrt(x(0)*x(0) + x(1)*x(1));
	for(i = 0; i < 20; i++)
	{ 
		y(y0 + s * rng::gaussian());
	}
}

void df(const arma::vec& x, running_stat<double> & y, arma::vec& g, parameters_t &p)
{
	unsigned int i;
	double y0, buf, s = 1e-2;
	
	y0 = 1 + sin(x(0)) * sin(x(1)) / sqrt(x(0)*x(0) + x(1)*x(1));
	for(i = 0; i < 20; i++)
	{
		y(y0 + s * rng::gaussian());
	}
	
	g.set_size(2);
	g(0) = cos(x(0)) * sin(x(1)) / sqrt(x(0)*x(0) + x(1)*x(1)) - x(0) * sin(x(0)) * sin(x(1)) / pow(x(0)*x(0) + x(1)*x(1), 1.5);
	g(1) = cos(x(1)) * sin(x(0)) / sqrt(x(0)*x(0) + x(1)*x(1)) - x(1) * sin(x(0)) * sin(x(1)) / pow(x(0)*x(0) + x(1)*x(1), 1.5);
}

void test_minimization_gradient()
{
	parameters_t p;
	arma::vec x;
	running_stat<double> v;
	
	x << 3.9 << 4.4;
	conjugate_gradient<parameters_t>::minimize(x, v, 0.3, 0.4, f, df, p);
}