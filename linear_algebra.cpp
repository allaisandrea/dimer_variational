#include "linear_algebra.h"
#include <exception>

extern "C"{
	
	void dgeqp3_(int*, int*, double*, int* m, int*, double*, double*, int*, int*);
	void dorgqr_(int *, int*, int*, double*, int*, double*, double*, int*, int*);
	void zgeqp3_(int*, int*, double*, int* m, int*, double*, double*, int*, double*, int*);
	void zungqr_(int *, int*, int*, double*, int*, double*, double*, int*, int*);
	void dger_(const int*, const int*, const double*, const double*, const int *, const double*, const int*, double*, const int*);
	void zgeru_(const int*, const int*, const arma::cx_double*, const arma::cx_double*, const int *, const arma::cx_double*, const int*, arma::cx_double*, const int*);
	void dcopy_(const int*, const double*, const int*, double*, const int*);
	void zcopy_(const int*, const arma::cx_double*, const int*, arma::cx_double*, const int*);
	void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
	void zaxpy_(const int*, const arma::cx_double*, const arma::cx_double*, const int*, arma::cx_double*, const int*);
	double ddot_(const int *, const double *, const int *, const double *, const int *);
	arma::cx_double zdotu_(const int *, const arma::cx_double *, const int *, const arma::cx_double *, const int *);
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


void rank_k_update(double a, const arma::Mat<double> &U, const arma::Mat<double>&V, arma::Mat<double>& A)
{
	int m, n, p, k, inc = 1;
	if(U.n_rows != A.n_rows || V.n_cols != A.n_cols || U.n_cols != V.n_rows)
		throw std::logic_error("Incorrect matrix dimensions");
	m = U.n_rows;
	n = V.n_cols;
	p = U.n_cols;
	for(k = 0; k < p; k++)
		dger_(&m, &n, &a, U.colptr(k), &inc, V.memptr() + k, &p, A.memptr(), &m); 

}

void rank_k_update(arma::cx_double a, const arma::Mat<arma::cx_double> &U, const arma::Mat<arma::cx_double>&V, arma::Mat<arma::cx_double>& A)
{
	int m, n, p, k, inc = 1;
	if(U.n_rows != A.n_rows || V.n_cols != A.n_cols || U.n_cols != V.n_rows)
		throw std::logic_error("Incorrect matrix dimensions");
	m = U.n_rows;
	n = V.n_cols;
	p = U.n_cols;
	for(k = 0; k < p; k++)
		zgeru_(&m, &n, &a, U.colptr(k), &inc, V.memptr() + k, &p, A.memptr(), &m); 

}

// template <class type>
// type trace_of_product(const arma::Mat<type> &_M1, const arma::Mat<type> &_M2)
// {
// 	unsigned int i, ni, j, m, n;
// 	type buf;
// 	const type *M1, *M2;
// 	
// 	m = _M1.n_rows;
// 	n = _M1.n_cols;
// 	if(_M2.n_rows != n || _M2.n_cols != m)
// 		throw std::logic_error("Incorrect matrix dimensions");
// 	
// 	M1 = _M1.memptr();
// 	M2 = _M2.memptr();
// 	
// 	buf = 0;
// 	for(i = 0; i < m; i++)
// 	{
// 		ni = n * i;
// 		for(j = 0; j < n; j++)
// 		{
// 			buf += M1[i + m * j] * M2[j + ni];
// 		}
// 	}
// 	return buf;
// }

double trace_of_product(const arma::Mat<double> &M1, const arma::Mat<double> &M2)
{
	int i, m, n, inc = 1;
	const double *_M1, *_M2;
	double buf;
	
	if(M2.n_rows != M1.n_cols || M2.n_cols != M1.n_rows)
		throw std::logic_error("Incorrect matrix dimensions");
	m = M1.n_rows;
	n = M1.n_cols;
	_M1 = M1.memptr();
	_M2 = M2.memptr();
	
	buf = 0.;
	for(i = 0; i < m; i++)
		buf += ddot_(&n, _M1 + i, & m, _M2 + n * i, &inc);
	return buf;
}

arma::cx_double trace_of_product(const arma::Mat<arma::cx_double> &M1, const arma::Mat<arma::cx_double> &M2)
{
	int i, m, n, inc = 1;
	const arma::cx_double *_M1, *_M2;
	arma::cx_double buf;
	
	if(M2.n_rows != M1.n_cols || M2.n_cols != M1.n_rows)
		throw std::logic_error("Incorrect matrix dimensions");
	m = M1.n_rows;
	n = M1.n_cols;
	_M1 = M1.memptr();
	_M2 = M2.memptr();
	
	buf = 0.;
	for(i = 0; i < m; i++)
		buf += zdotu_(&n, _M1 + i, & m, _M2 + n * i, &inc);
	return buf;
}

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


void copy_vector(int n, const double *x, int inc_x, double* y, int inc_y)
{
	dcopy_(&n, x, &inc_x, y, &inc_y);
}

void copy_vector(int n, const arma::cx_double *x, int inc_x, arma::cx_double* y, int inc_y)
{
	zcopy_(&n, x, &inc_x, y, &inc_y);
}

void add_to_vector(int n, double a, const double *x, int inc_x, double* y, int inc_y)
{
	daxpy_(&n, &a, x, &inc_x, y, &inc_y);
}

void add_to_vector(int n, arma::cx_double a, const arma::cx_double *x, int inc_x, arma::cx_double* y, int inc_y)
{
	zaxpy_(&n, &a, x, &inc_x, y, &inc_y);
}

template<class type>
void copy_vector_sparse(unsigned int n, const unsigned int *j, const type * x, unsigned int inc_x, type* y, unsigned int inc_y)
{
	unsigned int a, i;
	if(inc_x == 1)
	{
		i = 0;
		for(a = 0; a < n; a++)
		{
			y[i]  = x[(*j)];
			j++;
			i += inc_y;
		}
	}
	else
	{
		i = 0;
		for(a = 0; a < n; a++)
		{
			y[i]  = x[(*j) * inc_x];
			j++;
			i += inc_y;
		}
	}
}

template
void copy_vector_sparse<unsigned int>(unsigned int n, const unsigned int *j, const unsigned int * x, unsigned int inc_x, unsigned int* y, unsigned int inc_y);
template
void copy_vector_sparse<double>(unsigned int n, const unsigned int *j, const double * x, unsigned int inc_x, double* y, unsigned int inc_y);
template
void copy_vector_sparse<arma::cx_double>(unsigned int n, const unsigned int *j, const arma::cx_double * x, unsigned int inc_x, arma::cx_double* y, unsigned int inc_y);


template<class type>
void copy_matrix_sparse(unsigned int m, unsigned int n, const unsigned int *i, const unsigned int *j, const type * A, unsigned int ldA, type* B, unsigned int ldB)
{
	unsigned int a, b;
	const type* Aj;
	type* Bb;
	for(b = 0; b < n; b++)
	{
		Aj = A + ldA * j[b];
		Bb = B + ldB * b;
		for(a = 0; a < m; a++)
		{
			Bb[a]  = Aj[i[a]];
		}
	}
}

template
void copy_matrix_sparse<double>(unsigned int m, unsigned int n, const unsigned int *i, const unsigned int *j, const double * A, unsigned int ldA, double* B, unsigned int ldB);
template
void copy_matrix_sparse<arma::cx_double>(unsigned int m, unsigned int n, const unsigned int *i, const unsigned int *j, const arma::cx_double * A, unsigned int ldA, arma::cx_double* B, unsigned int ldB);