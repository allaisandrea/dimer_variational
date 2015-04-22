#ifndef __running_stat_h__
#define __running_stat_h__
#include <stdexcept>
#include <mpi.h>

class running_stat
{
public:
	friend void MPI_Send(running_stat& x, int dest, int tag, MPI_Comm comm);
	friend void MPI_Recv(running_stat& x, int source, int tag, MPI_Comm comm);
	
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
	running_stat & operator=(const running_stat &rs)
	{
		unsigned int i;
		reset(rs.nac);
		c = rs.c;
		m = rs.m;
		s2 = rs.s2;
		iac = rs.iac;
		for(i = 0; i < nac; i++)
		{
			ac[i] = rs.ac[i];
			dm_ac[i] = rs.dm_ac[i];
			yac[i] = rs.yac[i];
		}
		return *this;
	}
	void operator()(const double& x)
	{
		double dm;
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
		double dm;
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
	double mean() const
	{
		return m;
	}
	double second_moment() const
	{
		return s2;
	}
	double variance() const
	{
		if(c < 2)
			throw std::logic_error("Estimation of the variance requires at least two values.");
		return s2 * c / (c - 1);
	}
	double variance_of_the_mean() const
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
	double * ac, *dm_ac, *yac;
	
	unsigned int c, nac, iac;
	double m, s2;
};

inline void MPI_Send(running_stat& x, int dest, int tag, MPI_Comm comm)
{
	int err;
	
	err = MPI_Send(&x.nac, sizeof(x.nac), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(&x.c, sizeof(x.c), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
		
	err = MPI_Send(&x.m, sizeof(x.m), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(&x.s2, sizeof(x.s2), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(&x.iac, sizeof(x.iac), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.ac, x.nac * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.dm_ac, x.nac * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");

	err = MPI_Send(x.yac, x.nac * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
}

inline void MPI_Recv(running_stat& x, int source, int tag, MPI_Comm comm)
{
	int err;
	unsigned int nac;
	
	err = MPI_Recv(&nac, sizeof(nac), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	x.reset(nac);
	
	err = MPI_Recv(&x.c, sizeof(x.c), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
		
	err = MPI_Recv(&x.m, sizeof(x.m), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(&x.s2, sizeof(x.s2), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(&x.iac, sizeof(x.iac), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.ac, x.nac * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.dm_ac, x.nac * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");

	err = MPI_Recv(x.yac, x.nac * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
}
	
class gradient_running_stat
{
public:
	friend void MPI_Send(gradient_running_stat& x, int dest, int tag, MPI_Comm comm);
	friend void MPI_Recv(gradient_running_stat& x, int source, int tag, MPI_Comm comm);
	gradient_running_stat(unsigned int nd = 0)
	{
		reset(nd);
	};
	void reset(unsigned int nd = 0)
	{
		gradient_running_stat::nd = nd;
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
	
	void operator()(const gradient_running_stat &g)
	{
		if(g.nd != nd)
			throw std::logic_error("Dimensions do not match");
		F   = (n * F   + g.n * g.F  ) / (n + g.n);
		FF  = (n * FF  + g.n * g.FF ) / (n + g.n);
		Z   = (n * Z   + g.n * g.Z  ) / (n + g.n);
		FZ  = (n * FZ  + g.n * g.FZ ) / (n + g.n);
		FFZ = (n * FFZ + g.n * g.FFZ) / (n + g.n);
		ZZ  = (n * ZZ  + g.n * g.ZZ ) / (n + g.n);
		FZZ = (n * FZZ + g.n * g.FZZ) / (n + g.n);
		E   = (n * E   + g.n * g.E  ) / (n + g.n);
		n = n + g.n;
	}
	gradient_running_stat & operator=(const gradient_running_stat& g)
	{
		reset(g.nd);
		operator()(g);
		return *this;
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
	unsigned int n, nd;
	double F, FF;
	arma::vec Z, FZ, FFZ;
	arma::mat ZZ, FZZ, E;
};

inline void MPI_Send(gradient_running_stat& x, int dest, int tag, MPI_Comm comm)
{
	int err;
	
	err = MPI_Send(&x.nd, sizeof(x.nd), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(&x.n, sizeof(x.n), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(&x.F, sizeof(x.F), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(&x.FF, sizeof(x.FF), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.Z.memptr(), x.Z.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.FZ.memptr(), x.FZ.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.FFZ.memptr(), x.FFZ.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.ZZ.memptr(), x.ZZ.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.FZZ.memptr(), x.FZZ.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Send(x.E.memptr(), x.E.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
}


inline void MPI_Recv(gradient_running_stat& x, int source, int tag, MPI_Comm comm)
{
	int err;
	unsigned int nd;
	
	err = MPI_Recv(&nd, sizeof(nd), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	x.reset(nd);
	
	err = MPI_Recv(&x.n, sizeof(x.n), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(&x.F, sizeof(x.F), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(&x.FF, sizeof(x.FF), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.Z.memptr(), x.Z.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.FZ.memptr(), x.FZ.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.FFZ.memptr(), x.FFZ.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.ZZ.memptr(), x.ZZ.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.FZZ.memptr(), x.FZZ.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
	
	err = MPI_Recv(x.E.memptr(), x.E.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS) throw std::runtime_error("MPI_Send failed");
}

#endif
