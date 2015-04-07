#ifndef __minimization_h__
#define __minimization_h__
#include <armadillo>

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
	
	while(fabs(x) < 4.)
	{
		if(s1 > s2)
			f(x1, v1, param);
		else
			f(x2, v2, param);
		
		s1 = v1.variance_of_the_mean();
		s2 = v2.variance_of_the_mean();
		x = (v1.mean() - v2.mean()) / sqrt(s1 + s2);
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
	unsigned int it, max_it = 20, n_dim, i;
	int icomp, icomp1;
	unsigned int i_low, i_high;
	arma::vec sum, try_point, try_point1;
	running_stat<double> try_value, try_value1;
	arma::field<arma::vec> points;
	arma::field<running_stat<double> > values;
	arma::cube p_save;
	
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
	
	p_save.zeros(n_dim, n_dim + 1, max_it);
	for(it = 0; it < max_it; it++)
	{
		std::cout << "Find lowest and highest.\n";
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
		
		
		try_value.reset();
		try_point = simplex_reflect(-1., sum, points[i_high]);
		std::cout << "try:\n" << try_point << "\n";
		
		std::cout << "try < lowest?\n";
		icomp = compare(try_point, try_value, points[i_low], values[i_low], f, param);
		if(icomp < 0)
		{
			std::cout << "Yes.\n";
			try_value1.reset();
			try_point1 = simplex_reflect(-2., sum, points[i_high]);
			std::cout << "try1: " << trans(try_point1) << "";
			std::cout << "try1 < try?\n";
			icomp1 = compare(try_point1, try_value1, try_point, try_value, f, param);
			if(icomp1 < 0)
			{
				std::cout << "Yes. Expanding.\n";
				sum += try_point1 - points[i_high];
				points[i_high] = try_point1;
				values[i_high] = try_value1;
			}
			else
			{
				std::cout << "No. Reflecting.\n";
				sum += try_point - points[i_high];
				points[i_high] = try_point;
				values[i_high] = try_value;
			}
		}
		else
		{
			std::cout << "No.\n";
			std::cout << "try < highest?\n";
			icomp = compare(try_point, try_value, points[i_high], values[i_high], f, param);
			if(icomp < 0)
			{
				std::cout << "Yes. Reflecting\n";
				sum += try_point - points[i_high];
				points[i_high] = try_point;
				values[i_high] = try_value;
			}
			else
			{
				std::cout << "No.\n";
				try_value1.reset();
				try_point1 = simplex_reflect(0.5, sum, points[i_high]);
				std::cout << "try1: " << trans(try_point1) << "";
				std::cout << "try1 < highest?\n";
				icomp1 = compare(try_point1, try_value1, points[i_high], values[i_high], f, param);
				if(icomp1 < 0)
				{
					std::cout << "Yes. Contracting 1\n";
					sum += try_point1 - points[i_high];
					points[i_high] = try_point1;
					values[i_high] = try_value1;
				}
				else
				{
					std::cout << "No. Contracting all\n";
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
			p_save.slice(it).col(i) = points[i];
		p_save.save("p.bin");
	}
}

#endif
