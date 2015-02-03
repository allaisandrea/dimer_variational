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
	if(info != 0) std::cout << "QR decomposition failed" << "\n";
	
	lwork = buf;
	if(lwork > work.n_elem)
		work.set_size(lwork);

	dgeqp3_(&m, &n, A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) std::cout << "QR decomposition failed" << "\n";
	
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
	if(info != 0) std::cout << "Unitary Q reconstruction failed" << "\n";
}

void pivoting_qr_decomposition(arma::cx_mat &A, arma::cx_mat &Q, arma::uvec &perm)
{
	int i, j, m, n, lda, lwork, ltau, info,  tmp;
	double buf;
	static arma::vec work, tau;
	static arma::ivec jpvt;
	m = A.n_rows;
	n = A.n_cols;
	ltau = n < m ? n : m;
	jpvt.zeros(n);
	tau.set_size(2 * ltau);
	
	for(i = 0; i < n; i++)
		jpvt[i] = 0;
	
	lwork = -1;
	dgeqp3_(&m, &n, (double*)A.memptr(), &m, jpvt.memptr(), tau.memptr(), &buf, &lwork, &info);
	if(info != 0) std::cout << "QR decomposition failed" << "\n";
	
	lwork = 2 * buf;
	if(lwork > work.n_elem)
		work.set_size(lwork);

	dgeqp3_(&m, &n, (double*)A.memptr(), &m, jpvt.memptr(), tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) std::cout << "QR decomposition failed" << "\n";
	
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
	
	dorgqr_(&m, &m, &ltau, (double*)Q.memptr(), &m, tau.memptr(), work.memptr(), &lwork, &info);
	if(info != 0) std::cout << "Unitary Q reconstruction failed" << "\n";
}

template<class type>
bool singular(const arma::Mat<type> &M)
{
	arma::mat Q, R;
	arma::uvec perm;
	double max, min;
	R = M;
	pivoting_qr_decomposition(R, Q, perm);
	max = arma::max(abs(R.diag()));
	min = arma::min(abs(R.diag()));
	if(min == 0. || max / min > 1.e7)
		return true;
	return false;
}

template bool singular<double>(const arma::Mat<double> &M);
template bool singular<arma::cx_double>(const arma::Mat<arma::cx_double> &M);