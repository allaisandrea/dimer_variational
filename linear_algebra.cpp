#include "linear_algebra.h"
#include <exception>

extern "C"{
	void dgeqp3_(int*, int*, double*, int* m, int*, double*, double*, int*, int*);
	void dorgqr_(int *, int*, int*, double*, int*, double*, double*, int*, int*);
	void zgeqp3_(int*, int*, double*, int* m, int*, double*, double*, int*, double*, int*);
	void zungqr_(int *, int*, int*, double*, int*, double*, double*, int*, int*);
	void dger_(int*, int*, const double*, const double*, int *, const double*, int*, double*, int*);
	void zgeru_(int*, int*, const arma::cx_double*, const arma::cx_double*, int *, const arma::cx_double*, int*, arma::cx_double*, int*);
}

void pivoting_qr_decomposition(arma::mat &A, arma::mat &Q, arma::uvec &perm)
{
	int i, j, m, n, lda, lwork, ltau, info,  tmp;
	double buf;
	static arma::vec work, tau;
	static arma::ivec jpvt;
	m = A.n_rows;
	n = A.n_cols;
	ltau = n < m ? n : m;
	jpvt.zeros(n);
	tau.set_size(ltau);
	
	for(i = 0; i < n; i++)
		jpvt[i] = 0;
	
	lwork = -1;
	dgeqp3_(&m, &n, A.memptr(), &m, jpvt.memptr(), tau.memptr(), &buf, &lwork, &info);
	if(info != 0) throw std::runtime_error("QR decomposition failed");
	
	lwork = buf;
	if(lwork > work.n_elem)
		work.set_size(lwork);

	dgeqp3_(&m, &n, A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) throw std::runtime_error("QR decomposition failed");
	
	perm.set_size(n);
	for(i = 0; i < n; i++)
		perm[i] = jpvt[i] - 1;
		
	Q.zeros(m, m);
	for(j = 0; j < m; j++)
	for(i = j + 1; i < m; i++)
	{
		Q[i + m * j] = A[i + m * j];
		A[i + m * j] = 0.;
	}
	
	dorgqr_(&m, &m, &ltau, Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) throw std::runtime_error("Unitary Q reconstruction failed");
}

void pivoting_qr_decomposition(arma::cx_mat &A, arma::cx_mat &Q, arma::uvec &perm)
{
	int i, j, m, n, lda, lwork, ltau, info,  tmp;
	double buf;
	static arma::vec work, rwork, tau;
	static arma::ivec jpvt;
	m = A.n_rows;
	n = A.n_cols;
	ltau = n < m ? n : m;
	jpvt.zeros(n);
	tau.set_size(2 * ltau);
	rwork.set_size(2 * n);
	for(i = 0; i < n; i++)
		jpvt[i] = 0;
	
	lwork = -1;
	if(work.n_elem < 2 * (n + 1))
		work.set_size(2 * (n + 1));
	zgeqp3_(&m, &n, (double*)A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, rwork.memptr(), &info);
	if(info != 0) throw std::runtime_error("QR decomposition failed");
	
	lwork = 2 * work(0);
	if(lwork > work.n_elem)
		work.set_size(lwork);

	zgeqp3_(&m, &n, (double*)A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, rwork.memptr(), &info);
	if(info != 0) throw std::runtime_error("QR decomposition failed");
	
	perm.set_size(n);
	for(i = 0; i < n; i++)
		perm[i] = jpvt[i] - 1;
		
	Q.zeros(m, m);
	for(j = 0; j < m; j++)
	for(i = j + 1; i < m; i++)
	{
		Q[i + m * j] = A[i + m * j];
		A[i + m * j] = 0.;
	}
	
	zungqr_(&m, &m, &ltau, (double*)Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) throw std::runtime_error("Unitary Q reconstruction failed");
}


template<class type>
bool singular(const arma::Mat<type> &M)
{
	arma::Mat<type> Q, R;
	arma::vec s;
	arma::uvec perm;
	double max, min;
	R = M;
	pivoting_qr_decomposition(R, Q, perm);
	s = abs(R.diag());
	max = arma::max(s);
	min = arma::min(s);
	if(min == 0. || max / min > 1.e7)
		return true;
	return false;
}

template 
bool singular<double>(const arma::Mat<double> &M);
template 
bool singular<arma::cx_double>(const arma::Mat<arma::cx_double> &M);

void rank_1_update(double a, const arma::Col<double> &U, const arma::Row<double>&V, arma::Mat<double>& A)
{
	int m, n, inc = 1;
	if(U.n_rows != A.n_rows || V.n_cols != A.n_cols)
		throw std::logic_error("Incorrect matrix dimensions");
	m = U.n_rows;
	n = V.n_cols;
	
	dger_(&m, &n, &a, U.memptr(), &inc, V.memptr(), &inc, A.memptr(), &m); 
}

void rank_1_update(arma::cx_double a, const arma::Mat<arma::cx_double> &U, const arma::Mat<arma::cx_double>&V, arma::Mat<arma::cx_double>& A)
{
	int m, n, inc = 1;
	if(U.n_rows != A.n_rows || V.n_cols != A.n_cols)
		throw std::logic_error("Incorrect matrix dimensions");
	m = U.n_rows;
	n = V.n_cols;
	
	zgeru_(&m, &n, &a, U.memptr(), &inc, V.memptr(), &inc, A.memptr(), &m);
}


template<class type>
void rank_k_update(type a, const arma::Mat<type> &_U, const arma::Mat<type>&_V, arma::Mat<type>& _A)
{
	unsigned int i, j, nj, k, m, n, p, mk;
	type *A, aVj;
	const type *U, *V;
	
	m = _U.n_rows;
	n = _V.n_cols;
	p = _U.n_cols;
	if(_V.n_rows != p || _A.n_rows != m || _A.n_cols != n)
		throw std::logic_error("Incorrect matrix dimensions");
	
	
		
	U = _U.memptr();
	V = _V.memptr();
	A = _A.memptr();
	for(k = 0; k < p; k++)
	{
		for(j = 0; j < n; j++)
		{
			nj = n * j;
			aVj = a * V[k + p * j];
			mk = m * k;
			for(i = 0; i < m; i++)
				_A[i + nj] += _U[i + mk] * aVj;
		}
	}

}

template
void rank_k_update<double>(double a, const arma::Mat<double> &_U, const arma::Mat<double>&_V, arma::Mat<double>& _A);
template
void rank_k_update<arma::cx_double>(arma::cx_double a, const arma::Mat<arma::cx_double> &_U, const arma::Mat<arma::cx_double>&_V, arma::Mat<arma::cx_double>& _A);


template <class type>
type trace_of_product(const arma::Mat<type> &_M1, const arma::Mat<type> &_M2)
{
	unsigned int i, ni, j, m, n;
	type buf;
	const type *M1, *M2;
	
	m = _M1.n_rows;
	n = _M1.n_cols;
	if(_M2.n_rows != n || _M2.n_cols != m)
		throw std::logic_error("Incorrect matrix dimensions");
	
	M1 = _M1.memptr();
	M2 = _M2.memptr();
	
	buf = 0;
	for(i = 0; i < m; i++)
	{
		ni = n * i;
		for(j = 0; j < n; j++)
		{
			buf += M1[i + m * j] * M2[j + ni];
		}
	}
	return buf;
}

template 
double trace_of_product<double>(const arma::Mat<double> &_M1, const arma::Mat<double> &_M2);
template 
arma::cx_double trace_of_product<arma::cx_double>(const arma::Mat<arma::cx_double> &_M1, const arma::Mat<arma::cx_double> &_M2);


template <class type>
void eigensystem_variation(const arma::Mat<type> U, const arma::vec w, const arma::Mat<type> V, arma::Mat<type> &dU, arma::vec &dw)
{
	unsigned int i, j;
	double bandwidth;
	dU = trans(U) * V * U;
	dw = real(diagvec(dU));
	bandwidth = arma::max(w) - arma::min(w);
	for(i = 0; i < dU.n_rows; i++)
	{
		dU(i, i) = 0.;
		for(j = i + 1; j < dU.n_cols; j++)
		{
			if(fabs(w(i) - w(j)) / bandwidth > 1.e-8)
			{
				dU(i, j) /= w(j) - w(i);
				dU(j, i) /= w(i) - w(j);
			}
			else
			{
// 				if(abs(dU(i, j)) / bandwidth > 1.e-8 || abs(dU(j, i)) / bandwidth > 1.e-8)
// 					throw std::runtime_error("Perturbation does not share the symmetries of the hamiltonian");
				dU(i, j) = dU(j, i) = 0.;
			}
		}
	}
	
	dU = U * dU;
	
}

template
void eigensystem_variation<double>(const arma::Mat<double> U, const arma::vec w, const arma::Mat<double> V, arma::Mat<double> &dU, arma::vec &dw);
template
void eigensystem_variation<arma::cx_double>(const arma::Mat<arma::cx_double> U, const arma::vec w, const arma::Mat<arma::cx_double> V, arma::Mat<arma::cx_double> &dU, arma::vec &dw);


template<class type>
void copy_line(type c1, type c2, unsigned int m, unsigned int ldA, const type * A, unsigned int ldB, type* B)
{
	unsigned int a;

	if(c1 == (type) 1)
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  += A[a * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  -= A[a * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  +=  c2 * A[a * ldA];
			}
		}
	}
	else if(c1 == (type) 0)
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = A[a * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = -A[a * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c2 * A[a * ldA];
			}
		}
	}
	else if(c1 == (type) -1)
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = - B[a * ldB] + A[a * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = - B[a * ldB] - A[a * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = - B[a * ldB] + c2 * A[a * ldA];
			}
		}
	}
	else
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c1 * B[a * ldB] + A[a * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c1 * B[a * ldB] - A[a * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c1 * B[a * ldB] + c2 * A[a * ldA];
			}
		}
	}
}

template
void copy_line<unsigned int>(unsigned int c1, unsigned int c2, unsigned int m, unsigned int ldA, const unsigned int * A, unsigned int ldB, unsigned int* B);
template
void copy_line<double>(double c1, double c2, unsigned int m, unsigned int ldA, const double * A, unsigned int ldB, double* B);
template
void copy_line<arma::cx_double>(arma::cx_double c1, arma::cx_double c2, unsigned int m, unsigned int ldA, const arma::cx_double * A, unsigned int ldB, arma::cx_double* B);

template<class type>
void copy_line_sparse(type c1, type c2, unsigned int m, const unsigned int *j, unsigned int ldA, const type * A, unsigned int ldB, type* B)
{
	unsigned int a;

	if(c1 == (type) 1)
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  += A[j[a] * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  -= A[j[a] * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  +=  c2 * A[j[a] * ldA];
			}
		}
	}
	else if(c1 == (type) 0)
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = A[j[a] * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = -A[j[a] * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c2 * A[j[a] * ldA];
			}
		}
	}
	else if(c1 == (type) -1)
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = - B[a * ldB] + A[j[a] * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = - B[a * ldB] - A[j[a] * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = - B[a * ldB] + c2 * A[j[a] * ldA];
			}
		}
	}
	else
	{
		if(c2 == (type) 1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c1 * B[a * ldB] + A[j[a] * ldA];
			}
		}
		else if(c2 == (type) -1)
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c1 * B[a * ldB] - A[j[a] * ldA];
			}
		}
		else
		{
			for(a = 0; a < m; a++)
			{
				B[a * ldB]  = c1 * B[a * ldB] + c2 * A[j[a] * ldA];
			}
		}
	}
	
}

template
void copy_line_sparse<unsigned int>(unsigned int c1, unsigned int c2, unsigned int m, const unsigned int *j, unsigned int ldA, const unsigned int * A, unsigned int ldB, unsigned int* B);
template
void copy_line_sparse<double>(double c1, double c2, unsigned int m, const unsigned int *j, unsigned int ldA, const double * A, unsigned int ldB, double* B);
template
void copy_line_sparse<arma::cx_double>(arma::cx_double c1, arma::cx_double c2, unsigned int m, const unsigned int *j, unsigned int ldA, const arma::cx_double * A, unsigned int ldB, arma::cx_double* B);







