#ifndef __minimization_gradient_h__
#define __minimization_gradient_h__

#include <armadillo>
#include <vector>
#include "running_stat.h"
#include "utilities.h"
#include "rng.h"

template <class parameters_t>
class conjugate_gradient
{
public:
	typedef void (*df_pointer)(const arma::vec&, running_stat&, gradient_running_stat&, parameters_t &);
	
	static void minimize(
		arma::vec &x,
		running_stat  &v,
		double step,
		unsigned int max_count,
		unsigned int max_it,
		df_pointer df,
		parameters_t & param)
	{
		unsigned int i, j;
		gradient_running_stat gstat;
		
		param.points.zeros(x.n_rows, max_it);
		param.values.zeros(2, max_it);
		
		v.reset();
		gstat.reset(x.n_rows);
		while(v.count() < max_count)
			df(x, v, gstat, param);
		
		param.points.col(0) = x;
		param.values(0, 0) = v.mean();
		param.values(1, 0) = sqrt(v.variance_of_the_mean());
		
		std::cout << elapsed_time_string() << " x: " << trans(x);
		std::cout << elapsed_time_string() << " f: " << v.mean() << " +- " << sqrt(v.variance_of_the_mean()) << " @ " << v.count() << std::endl;
		std::cout << elapsed_time_string() << " g: " <<  norm(gstat.gradient(), 2) << " | " << trans(gstat.gradient());
		std::cout << elapsed_time_string() << " sqrt(g.1/ss.g): " << sqrt(dot(gstat.gradient(), inv(gstat.gradient_covariance()) * gstat.gradient())) << "\n" << std::endl;

		
		for(i = 1; i < max_it; i++)
		{
			line_search(x, v, gstat, gstat.gradient() / norm(gstat.gradient(), 2), step, max_count, df, param);
			while(v.count() < max_count)
				df(x, v, gstat, param);
			
			param.points.col(i) = x;
			param.values(0, i) = v.mean();
			param.values(1, i) = sqrt(v.variance_of_the_mean());
		
			std::cout << elapsed_time_string() << " x: " << trans(x);
			std::cout << elapsed_time_string() << " f: " << v.mean() << " +- " << sqrt(v.variance_of_the_mean()) << " @ " << v.count() << std::endl;
			std::cout << elapsed_time_string() << " g: " <<  norm(gstat.gradient(), 2) << " | " << trans(gstat.gradient());
			std::cout << elapsed_time_string() << " sqrt(g.1/ss.g): " << sqrt(dot(gstat.gradient(), inv(gstat.gradient_covariance()) * gstat.gradient())) << "\n" << std::endl;
			param.write();
		}
	}
	
protected:
	
	static void line_search(
		arma::vec &x,
		running_stat &v0,
		gradient_running_stat &g0,
		const arma::vec &G,
		double step,
		unsigned int max_count,
		df_pointer df,
		parameters_t & param)
	{
		unsigned int i0, i1, i2;
		int icomp;
		double r0, r1, r2, r3;
		std::vector<running_stat > v(4);
		std::vector<gradient_running_stat > g(4);
		
		i0 = 0; i1 = 1; i2 = 2;
		r0 = 0.; v[i0] = v0; g[i0] = g0;
		
		std::cout << elapsed_time_string() << " Line search\n";
		std::cout << elapsed_time_string() << " Bracketing\n";
		std::cout.flush();
		
		r1 = step * (0.5 + rng::uniform());
		v[i1].reset();
		g[i1].reset(x.n_rows);
		icomp = compare(x - r1 * G, v[i1], g[i1], x - r0 * G, v[i0], g[i0], max_count, df, param);
		if(icomp < 0)
		{
			std::cout << elapsed_time_string() << " Undershoot\n";
			std::cout.flush();
			
			r2 = r0;
			v[i2] = v[i0];
			g[i2] = g[i0];
			do{
				
				r0 = r2;
				v[i0] = v[i2];
				g[i0] = g[i2];
				r2 = 1.5 * r1;
				v[i2].reset();
				g[i2].reset(x.n_rows);
				icomp = compare(x - r2 * G, v[i2], g[i2], x - r1 * G, v[i1], g[i1], max_count, df, param);
				std::cout << elapsed_time_string() << std::setw(15) <<  r0 << " " << std::setw(15) << v[i0].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(15) <<  r1 << " " << std::setw(15) << v[i1].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(15) <<  r2 << " " << std::setw(15) << v[i2].mean() << "\n" << std::endl;
				std::cout.flush();
				swap(i1, i2);
				swap(r1, r2);
			}while(icomp < 0);
			swap(i1, i2);
			swap(r1, r2);
		}
		else if(icomp >= 0)
		{
			std::cout << elapsed_time_string() << " Overshoot\n";
			std::cout.flush();
			r2 = r1;
			v[i2] = v[i1];
			g[i2] = g[i1];
			do{
				r1 = r2;
				v[i1] = v[i2];
				g[i1] = g[i2];
				
				r2 = 0.5 * r2;
				v[i2].reset();
				g[i2].reset(x.n_rows);
				icomp = compare(x - r2 * G, v[i2], g[i2], x - r0 * G, v[i0], g[i0], max_count, df, param);
				std::cout << elapsed_time_string() << std::setw(15) << r0 << " " << std::setw(15) << v[i0].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(15) << r1 << " " << std::setw(15) << v[i1].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(15) << r2 << " " << std::setw(15) << v[i2].mean() << "\n" << std::endl;
			}while(icomp > 0);
			swap(i1, i2);
			swap(r1, r2);
		}
		
		std::cout << elapsed_time_string() << " Bracketed\n";
		std::cout << elapsed_time_string() << std::setw(15) << r0 << " " << std::setw(15) << v[i0].mean() << "\n";
		std::cout << elapsed_time_string() << std::setw(15) << r1 << " " << std::setw(15) << v[i1].mean() << "\n";
		std::cout << elapsed_time_string() << std::setw(15) << r2 << " " << std::setw(15) << v[i2].mean() << "\n" << std::endl;
		
		while(icomp != 0)
		{
			if(fabs((r2 - r1) / (r1 - r0)) < 1.)
			{
				swap(i0, i2);
				swap(r0, r2);
			}
			
			r3 = 0.5 * (r1 + r2);
			v[3].reset();
			g[3].reset(x.n_rows);
			icomp = compare(x - r3 * G, v[3], g[3], x - r1 * G, v[i1], g[i1], max_count, df, param);
			std::cout << elapsed_time_string() << std::setw(15) << r3 << " " << std::setw(15) << v[3].mean() << "\n";
			if(icomp <= 0)
			{
				swap(i0, i1);
				swap(r0, r1);
				r1 = r3;
				v[i1] = v[3];
				g[i1] = g[3];
			}
			else
			{
				r2 = r3;
				v[i2] = v[3];
				g[i2] = g[3];
			}
			
			std::cout << elapsed_time_string() << std::setw(15) << r0 << " " << std::setw(15) << v[i0].mean() << "\n";
			std::cout << elapsed_time_string() << std::setw(15) << r1 << " " << std::setw(15) << v[i1].mean() << "\n";
			std::cout << elapsed_time_string() << std::setw(15) << r2 << " " << std::setw(15) << v[i2].mean() << "\n" << std::endl;
		}
		
		x -= r1 * G;
		v0 = v[i1];
		g0 = g[i1];
	}
	
	static int compare(
		const arma::vec &x1, 
		running_stat &v1,
		gradient_running_stat &g1,
		const arma::vec &x2, 
		running_stat &v2,
		gradient_running_stat &g2,
		unsigned int max_count,
		df_pointer df,
		parameters_t &param)
	{
		double re, bre;
		
		arma::vec vbuf;
		if(v1.count() < 2)
			df(x1, v1, g1, param);
		if(v2.count() < 2)
			df(x2, v2, g2, param);
		
		re = (v1.mean() - v2.mean()) / sqrt(v1.variance() / v1.count() + v2.variance() / v2.count());
		bre = (v1.mean() - v2.mean()) / sqrt(v1.variance() / max_count + v2.variance() / max_count);
		std::cout << elapsed_time_string() << " \trelative diff: " << " " << std::setw(15) <<  re << " @ " << std::setw(8) << v1.count() << ", " << std::setw(8) << v2.count()<< std::endl;
		
		while(fabs(re) < 4. && (v1.count() < max_count || v2.count() < max_count))
		{
			if(v1.variance_of_the_mean() > v2.variance_of_the_mean())
				df(x1, v1, g1, param);
			else
				df(x2, v2, g2, param);
			
			re = (v1.mean() - v2.mean()) / sqrt(v1.variance() / v1.count() + v2.variance() / v2.count());
			bre = (v1.mean() - v2.mean()) / sqrt(v1.variance() / max_count + v2.variance() / max_count);
			std::cout << elapsed_time_string() << " \trelative diff: " << " " << std::setw(15) <<  re << " @ "<< std::setw(8) << v1.count() << ", " << std::setw(8) << v2.count()<< std::endl;
		}
		
		if(fabs(re) < 4)
			return 0;
		else if(re < 0.)
			return -1;
		else
			return +1;
	}
};

#endif
