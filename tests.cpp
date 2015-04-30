#include "states.h"

unsigned int basis_functions(unsigned int L, unsigned int k, unsigned int q, arma::mat& psi);
arma::mat homogeneous_state_hamiltonian(
	unsigned int L,
	double dmu, 
	double t1, 
	double t2, 
	double t3, 
	double t4);

arma::mat homogeneous_state_hamiltonian(unsigned int L, const arma::vec & u);

void kspace_hamiltonian(const arma::vec &u, unsigned int L, unsigned int ik, unsigned int iq, arma::mat& h, arma::cube& dh);

void test_basis_functions()
{
	unsigned int L = 6, L_2 = L / 2, k, q, i;
	arma::mat psi, H, h;
	arma::cube dh;
	arma::vec u;
	
	u.randn(6);
	H = homogeneous_state_hamiltonian(L, u);
	
	i = 0;
	for(q = 0; q <= L_2; q++)
	for(k = 0; k < L; k++)
	{
		kspace_hamiltonian(u, L, k, q, h, dh);
		
		basis_functions(L, k, q, psi);
		
		std::cout << std::fixed << std::setprecision(5) << std::setw(10);
		
		(trans(psi) * psi).raw_print();
		std::cout << "\n"; 
		
		std::cout << std::setw(10);
		(trans(psi) * H * psi).raw_print();
		std::cout << "\n";
		
		std::cout << std::setw(10);
		h.raw_print();
		std::cout << "================\n";
	}
	
	
}

void test_homogeneous_state()
{
	unsigned int L = 8, L_2 = L / 2, k, q, nf, i;
	double beta = 10, eps = 1.e-7;
	data_structures<double> ds;
	arma::vec x, w1, w2, Dw, u0;
	arma::mat H, P, Dpsi, psi1, psi2;
	
	ds.L = L;
	build_graph(ds);
	
	P.randn(6, 4);
	x.randn(4);
	u0.zeros(6);
	
	for(i = 0; i < x.n_rows; i++)
	{
	
		x(i) += eps;
		homogeneous_state(x, P, u0, beta, ds);
		psi1 = ds.psi[0];
		w1 = ds.w[0];
		
		x(i) -= 2 * eps;
		homogeneous_state(x, P, u0, beta, ds);
		psi2 = ds.psi[0];
		w2 = ds.w[0];
		
		Dpsi = (psi1 - psi2) / eps / 2;
		Dw = (w1 - w2) / eps / 2;
		
		x(i) += eps;
		homogeneous_state(x, P, u0, beta, ds);
		
// 		std::cout << join_rows(ds.Dw[0].col(i),  Dw) << "\n";
		std::cout << norm(ds.Dw[0].col(i) - Dw, "fro")  << "\n";
		std::cout << norm(ds.Dpsi[0].slice(i) - Dpsi, "fro")  << "\n\n";
	}
	
	H = homogeneous_state_hamiltonian(L, P * x);
	ds.psi[0].save("U.bin");
	H.save("H.bin");
	ds.w[0].save("w.bin");
}

