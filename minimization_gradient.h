#ifndef __minimization_gradient_h__
#define __minimization_gradient_h__

#include <armadillo>
#include <vector>
#include "running_stat.h"
#include "utilities.h"

template <class parameters_t>
class conjugate_gradient
{
public:
	typedef void (*f_pointer)(const arma::vec&, running_stat<double>&, parameters_t &);
	typedef void (*df_pointer)(const arma::vec&, running_stat<double>&, arma::vec&, parameters_t &);
	
	static void minimize(
		arma::vec &x,
		running_stat <double> &v,
		double step,
		double accuracy,
		f_pointer f,
		df_pointer df,
		parameters_t & param)
	{
		unsigned int i;
		arma::vec h, g, new_g;
		double norm_g, norm_new_g, delta_g_dot_new_g;
		
		v.reset();
		df(x, v, g, param);
		h = g;
		norm_g = norm(g, 2);
		arma::mat xx(x.n_rows, 6);
		xx.col(0) = x;
		for(i = 0; i < 5; i++)
		{
			line_search(x, v, h / norm(h, 2), step, accuracy, f, param);
			df(x, v, new_g, param);
			xx.col(i + 1) = x;
			norm_new_g = dot(new_g, new_g);
			delta_g_dot_new_g = norm_new_g - dot(g, new_g);

			h = new_g + delta_g_dot_new_g / norm_g * h;
			g = new_g;
			norm_g = norm_new_g;
			
			h *= dot(h, g) / fabs(dot(h, g));
			
			xx.save("x.bin");
		}
	}
	
protected:
	
	static void line_search(
		arma::vec &x,
		running_stat<double> &v0,
		const arma::vec &G,
		double step,
		double accuracy,
		f_pointer f,
		parameters_t & param)
	{
		unsigned int i0, i1, i2;
		int icomp;
		double r0, r1, r2, r3;
		std::vector<running_stat<double> > v(4);
		
		
		i0 = 0; i1 = 1; i2 = 2;
		r0 = 0.; v[i0] = v0;
		
		std::cout << elapsed_time_string() << " Line search\n";
		std::cout << elapsed_time_string() << " Bracketing\n";
		std::cout.flush();
		
		r1 = step;
		v[i1].reset();
		icomp = compare(x - r1 * G, v[i1], x - r0 * G, v[i0], f, param);
		if(icomp < 0)
		{
			std::cout << elapsed_time_string() << " Undershoot\n";
			std::cout.flush();
			
			r2 = r0;
			v[i2] = v[i0];
			do{
				
				r0 = r2;
				v[i0] = v[i2];
				
				r2 = 1.5 * r1;
				v[i2].reset();
				icomp = compare(x - r2 * G, v[i2], x - r1 * G, v[i1], f, param);
				std::cout << elapsed_time_string() << std::setw(8) <<  r0 << " " << v[i0].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(8) <<  r1 << " " << v[i1].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(8) <<  r2 << " " << v[i2].mean() << "\n" << std::endl;
				std::cout.flush();
				swap(i1, i2);
				swap(r1, r2);
			}while(icomp < 0);
			swap(i1, i2);
			swap(r1, r2);
			
		}
		else
		{
			std::cout << elapsed_time_string() << " Overshoot\n";
			std::cout.flush();
			r2 = r1;
			v[i2] = v[i1];
			do{
				r1 = r2;
				v[i1] = v[i2];
				
				r2 = 0.5 * r2;
				v[i2].reset();
				icomp = compare(x - r2 * G, v[i2], x - r0 * G, v[i0], f, param);
				std::cout << elapsed_time_string() << std::setw(8) << r0 << " " << v[i0].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(8) <<  r1 << " " << v[i1].mean() << "\n";
				std::cout << elapsed_time_string() << std::setw(8) <<  r2 << " " << v[i2].mean() << "\n" << std::endl;
			}while(icomp > 0);
		}
		
		std::cout << elapsed_time_string() << "Bracketed\n";
		std::cout << elapsed_time_string() << std::setw(8) <<  r0 << " " << v[i0].mean() << "\n";
		std::cout << elapsed_time_string() << std::setw(8) <<  r1 << " " << v[i1].mean() << "\n";
		std::cout << elapsed_time_string() << std::setw(8) <<  r2 << " " << v[i2].mean() << "\n" << std::endl;
				
		while(fabs(r2 - r0) > accuracy)
		{
			if(fabs((r2 - r1) / (r1 - r0)) < 1.)
			{
				swap(i0, i2);
				swap(r0, r2);
			}
			
			r3 = 0.5 * (r1 + r2);
			v[3].reset();
			icomp = compare(x - r3 * G, v[3], x - r1 * G, v[i1], f, param);
			std::cout << elapsed_time_string() << std::setw(8) <<  r3 << " " << v[3].mean() << "\n" << std::endl;
			if(icomp < 0)
			{
				swap(i0, i1);
				swap(r0, r1);
				r1 = r3;
				v[i1] = v[3];
			}
			else
			{
				r2 = r3;
				v[i2] = v[3];
			}
			
			std::cout << elapsed_time_string() << std::setw(8) <<  r0 << " " << v[i0].mean() << "\n";
			std::cout << elapsed_time_string() << std::setw(8) <<  r1 << " " << v[i1].mean() << "\n";
			std::cout << elapsed_time_string() << std::setw(8) <<  r2 << " " << v[i2].mean() << "\n" << std::endl;
		}
		
		x -= r1 * G;
		v0 = v[i1];
	}
	
	static int compare(
		const arma::vec &x1, 
		running_stat<double> &v1, 
		const arma::vec &x2, 
		running_stat<double> &v2, 
		f_pointer f, 
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
// 		std::cout << "\t" << v1.count() << " " << v2.count() << " " <<  x << std::endl;
		
		while(fabs(x) < 4.)
		{
			if(s1 > s2)
				f(x1, v1, param);
			else
				f(x2, v2, param);
			
			s1 = v1.variance_of_the_mean();
			s2 = v2.variance_of_the_mean();
			x = (v1.mean() - v2.mean()) / sqrt(s1 + s2);
// 			std::cout << " \t" << v1.count() << " " << v2.count() << " " <<  x << std::endl;
		}
		
		std::cout << elapsed_time_string() << " \tcomparison:\n";
		std::cout << elapsed_time_string() << " \t" <<  v1.count()  << " " << v1.mean() << " " << sqrt(v1.variance_of_the_mean()) << trans(x1);
		std::cout << elapsed_time_string() << " \t" <<  v1.count()  << " " << v2.mean() << " " << sqrt(v1.variance_of_the_mean()) << trans(x2);
		std::cout.flush();
		if(x < 0.)
			return -1;
		else
			return +1;
	}
};

#endif
