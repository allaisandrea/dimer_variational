#include "linear_algebra.h"
extern "C"{
	void dgeqp3_(int*, int*, double*, int* m, int*, double*, double*, int*, int*);
	void dorgqr_(int *, int*, int*, double*, int*, double*, double*, int*, int*);
	void zgeqp3_(int*, int*, double*, int* m, int*, double*, double*, int*, double*, int*);
	void zungqr_(int *, int*, int*, double*, int*, double*, double*, int*, int*);
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
	if(info != 0) throw "QR decomposition failed";
	
	lwork = buf;
	if(lwork > work.n_elem)
		work.set_size(lwork);

	dgeqp3_(&m, &n, A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) throw "QR decomposition failed";
	
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
	if(info != 0) throw "Unitary Q reconstruction failed";
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
	if(info != 0) throw "QR decomposition failed";
	
	lwork = 2 * work(0);
	if(lwork > work.n_elem)
		work.set_size(lwork);

	zgeqp3_(&m, &n, (double*)A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, rwork.memptr(), &info);
	if(info != 0) throw "QR decomposition failed";
	
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
	if(info != 0) throw "Unitary Q reconstruction failed";
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

template bool singular<double>(const arma::Mat<double> &M);
template bool singular<arma::cx_double>(const arma::Mat<arma::cx_double> &M);