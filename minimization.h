#ifndef __minimization_h__
#define __minimization_h__
#include <armadillo>
#include <ctime>
#include "utilities.h"

extern time_t start_time;

template <class parameters_t>
int compare(
	const arma::vec &x1, 
	running_stat<double> &v1, 
	const arma::vec &x2, 
	running_stat<double> &v2, 
	void (*f)(
		const arma::vec&, 
		running_stat<double>&, 
		parameters_t &), 
	parameters_t &param)
{
	double x, s1, s2;
	
	arma::vec vbuf;
	if(v1.count() < 2)
		f(x1, v1, param);
	if(v2.count() < 2)
		f(x2, v2, param);
	
	s1 = v1.variance_of_the_mean();
	s2 = v2.variance_of_the_mean();
	x = (v1.mean() - v2.mean()) / sqrt(s1 + s2);
	std::cout << elapsed_time_string() << " \t" << v1.count() << " " << v2.count() << " " <<  x << std::endl;
	
	while(fabs(x) < 4.)
	{
		if(s1 > s2)
			f(x1, v1, param);
		else
			f(x2, v2, param);
		
		s1 = v1.variance_of_the_mean();
		s2 = v2.variance_of_the_mean();
		x = (v1.mean() - v2.mean()) / sqrt(s1 + s2);
		std::cout << elapsed_time_string() << " \t" << v1.count() << " " << v2.count() << " " <<  x << std::endl;
	}
	
	if(x < 0.)
		return -1;
	else
		return +1;
}


arma::vec simplex_reflect(double x, const arma::vec &sum, arma::vec & point)
{
	double x1, x2;
	x1 = (1. - x) / point.n_elem;
	x2 = x1 - x;
	return sum * x1 - point * x2;
}

template <class parameters_t>
void simplex_minimize(
	arma::mat &_points, 
	void (*f)(
		const arma::vec&, 
		running_stat<double>&, 
		parameters_t &), 
	parameters_t & param)
{
	unsigned int it, max_it = 20, n_dim, i, j;
	int icomp, icomp1;
	unsigned int i_low, i_high;
	arma::vec sum, try_point, try_point1;
	running_stat<double> try_value, try_value1;
	arma::field<arma::vec> points;
	arma::field<running_stat<double> > values;
	
	n_dim = _points.n_rows;
	if(_points.n_cols != n_dim + 1)
		throw std::logic_error("Wrong number of corners for a simplex");
	
	values.set_size(n_dim + 1);
	points.set_size(n_dim + 1);
	sum.zeros(n_dim);
	for(i = 0; i < n_dim + 1; i++)
	{
		points[i] = _points.col(i);
		sum += points[i];
	}
	
	param.simplexes.zeros((n_dim + 2) * (n_dim + 1), max_it);
	for(it = 0; it < max_it; it++)
	{
		std::cout << elapsed_time_string() << " Find lowest and highest." << std::endl;
		i_low = i_high = 0;
		for(i = 1; i < n_dim + 1; i++)
		{
			icomp = compare(points[i], values[i], points[i_low], values[i_low], f, param);
			if(icomp < 0)
				i_low = i;
			
			icomp = compare(points[i], values[i], points[i_high], values[i_high], f, param);
			if(icomp > 0)
				i_high = i;
		}
		
		std::cout << elapsed_time_string() << " lowest:";
		std::cout << std::fixed << std::setprecision(6);
		std::cout << std::setw(10) << values[i_low].mean();
		std::cout << std::setw(10) << sqrt(values[i_low].variance_of_the_mean());
		std::cout << std::endl;
		
		try_value.reset();
		try_point = simplex_reflect(-1., sum, points[i_high]);
		std::cout << elapsed_time_string() << " try:" << trans(try_point);
		
		std::cout << elapsed_time_string() << " try < lowest?" << std::endl;
		icomp = compare(try_point, try_value, points[i_low], values[i_low], f, param);
		if(icomp < 0)
		{
			std::cout << elapsed_time_string() << " Yes." << std::endl;
			try_value1.reset();
			try_point1 = simplex_reflect(-2., sum, points[i_high]);
			std::cout << elapsed_time_string() << " try1: " << trans(try_point1);
			std::cout << elapsed_time_string() << " try1 < try?" << std::endl;
			icomp1 = compare(try_point1, try_value1, try_point, try_value, f, param);
			if(icomp1 < 0)
			{
				std::cout << elapsed_time_string() << " Yes. Expanding." << std::endl;
				sum += try_point1 - points[i_high];
				points[i_high] = try_point1;
				values[i_high] = try_value1;
			}
			else
			{
				std::cout << elapsed_time_string() << "No. Reflecting." << std::endl;
				sum += try_point - points[i_high];
				points[i_high] = try_point;
				values[i_high] = try_value;
			}
		}
		else
		{
			std::cout << elapsed_time_string() << " No." << std::endl;
			std::cout << elapsed_time_string() << " try < highest?" << std::endl;
			icomp = compare(try_point, try_value, points[i_high], values[i_high], f, param);
			if(icomp < 0)
			{
				std::cout << elapsed_time_string()<< " Yes. Reflecting" << std::endl;
				sum += try_point - points[i_high];
				points[i_high] = try_point;
				values[i_high] = try_value;
			}
			else
			{
				std::cout << elapsed_time_string() << " No." << std::endl;
				try_value1.reset();
				try_point1 = simplex_reflect(0.5, sum, points[i_high]);
				std::cout << elapsed_time_string() << " try1: " << trans(try_point1);
				std::cout << elapsed_time_string() << " try1 < highest?" << std::endl;
				icomp1 = compare(try_point1, try_value1, points[i_high], values[i_high], f, param);
				if(icomp1 < 0)
				{
					std::cout << elapsed_time_string() << " Yes. Contracting 1" << std::endl;
					sum += try_point1 - points[i_high];
					points[i_high] = try_point1;
					values[i_high] = try_value1;
				}
				else
				{
					std::cout << elapsed_time_string() << " No. Contracting all" << std::endl;
					sum.zeros();
					for(i = 0; i < n_dim + 1; i++)
					{
						if(i != i_low)
						{
							points[i] = 0.5 * (points[i] + points[i_low]);
							values[i].reset();
						}
						sum+=points[i];
					}
				}
			}
		}
		for(i = 0; i < n_dim + 1; i++)
		{
			for(j = 0; j < n_dim; j++)
				param.simplexes(j + (n_dim + 2) * i, it) = points[i](j);
			param.simplexes(j + (n_dim + 2) * i, it) = values[i].mean();
			j++;
			param.simplexes(j + (n_dim + 2) * i, it) = sqrt(values[i].variance_of_the_mean());
		}
		param.write();
	}
}

#endif
