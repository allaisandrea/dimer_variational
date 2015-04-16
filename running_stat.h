#ifndef __running_stat_h__
#define __running_stat_h__
#include <stdexcept>

template<class type>
class running_stat
{
public:
	running_stat(unsigned int nac = 0)
	{
		ac = 0x0;
		yac = 0x0;
		dm_ac = 0x0;
		reset(nac);
	}
	void reset(unsigned int nac = 0)
	{
		running_stat::nac = nac;
		iac = 0;
		c = 0;
		m = s2 = 0.;
		if(nac > 0)
		{
			unsigned int i;
			if(ac) delete[] ac;
			if(yac) delete[] yac;
			if(dm_ac) delete[] dm_ac;
			ac = new double[nac];
			yac = new double[nac];
			dm_ac = new double[nac];
			for(i = 0; i < nac; i++)
			{
				ac[i] = yac[i] = dm_ac[i] = 0.;
			}
		}
	}
	void operator()(const type& x)
	{
		type dm;
		double dc, dc1;
		
		dc1 = c;
		c++;
		dc = c;
		
		dm = (x - m) / dc;
		m += dm;
		s2 += dc1 * dm * dm - s2 / dc;
		
		if(nac > 0)
		{
			unsigned int i;
			dm_ac[iac % nac] = dm;
			for(i = 0; i < (nac < c ? nac : c); i++)
			{
				dm = (x - yac[i]) / (c - i);
				yac[i] += dm;
				ac[i] += (c - i - 1) * dm_ac[(nac + iac - i) % nac] * dm - ac[i] / (c - i);
			}
			iac++;
		}
	}
	void operator()(const running_stat& rs)
	{
		type dm;
		double dc1, dc2, dc12;
		
		dc1 = c;
		dc2 = rs.count();
		dc12 = c + rs.count();
		dc12 /= dc2;
		
		c += rs.count();
		
		dm = (rs.mean() - m) / dc12;
		m += dm;
		s2 += (rs.second_moment() - s2) / dc12 + dm * dm * dc1 / dc2;
	}
	unsigned int count() const
	{
		return c;
	}
	type mean() const
	{
		return m;
	}
	type second_moment() const
	{
		return s2;
	}
	type variance() const
	{
		if(c < 2)
			throw std::logic_error("Estimation of the variance requires at least two values.");
		return s2 * c / (c - 1);
	}
	type variance_of_the_mean() const
	{
		if(c < 2)
			throw std::logic_error("Estimation of the variance requires at least two values.");
		return s2 / (c - 1);
	}
	arma::vec autocorrelation() const
	{
		unsigned int i;
		arma::vec res(ac, nac);
		for(i = 0; i < nac; i++)
		{
			res[i] *= (c - i) / (c - i - 1.);
		}
		return res;
	}
protected:
	type * ac, *dm_ac, *yac;
	
	unsigned int c, nac, iac;
	type m, s2;
};

class gradient_running_stat
{
public:
	gradient_running_stat(unsigned int nd = 0)
	{
		reset(nd);
	};
	void reset(unsigned int nd = 0)
	{
		n = 0;
		F = 0.;
		FF = 0.;
		Z.zeros(nd);
		FZ.zeros(nd);
		FFZ.zeros(nd);
		ZZ.zeros(nd, nd);
		FZZ.zeros(nd, nd);
		E.zeros(nd, nd);
	}
	void operator()(double Fi, const arma::vec &Zi)
	{
		double dF, dFF, n0, n1;
		static arma::vec dZ, dFZ, dFFZ;
		static arma::mat dZZ, dFZZ, dE;
		n0 = n;
		n++;
		n1 = n;
		
		dF = (Fi - F) / n1;
		F += dF;
		dZ = (Zi - Z) / n1;
		Z += dZ;
		
		dFF = n0 * dF * dF - FF / n1;
		FF += dFF;
		dFZ = n0 * dF * dZ - FZ / n1;
		FZ += dFZ;
		dZZ = n0 * kron(trans(dZ), dZ) - ZZ / n1;
		ZZ += dZZ;
		
		dFFZ = n0 * (2 * dF * dFZ + dZ * dFF - (2 * n0 + 1) * dF * dF * dZ) - FFZ / n1;
		FFZ += dFFZ;
		
		dFZZ = n0 * (
			dF * dZZ 
			+ kron(trans(dZ), dFZ) 
			+ kron(trans(dFZ), dZ) 
			- (2 * n0 + 1) * dF * kron(trans(dZ), dZ)) - FZZ / n1;
		FZZ += dFZZ;
		
		dE = n0 * (
				2 * dF * dFZZ
				+ kron(trans(dZ), dFFZ) 
				+ kron(trans(dFFZ), dZ)
				+ kron(trans(dFZ), dFZ)
				- (2 * n0 + 1) * (
					2 * dF * kron(trans(dZ), dFZ)
					+ 2 * dF * kron(trans(dFZ), dZ)
					+ dF * dF * dZZ
					+ dFF * kron(trans(dZ), dZ))
				+ (8 * n0 * n0 + 8 * n0 + 1) * dF * dF * kron(trans(dZ), dZ))
			- E / n1;
		E += dE;
		
	}
	unsigned int count()
	{
		return n;
	}
	
	arma::vec gradient()
	{
		return FZ * n / (n - 1);
	}
	arma::mat gradient_covariance()
	{
		return E / n;
	}
protected:
	unsigned int n;
	double F, FF;
	arma::vec Z, FZ, FFZ;
	arma::mat ZZ, FZZ, E;
};
#endif
