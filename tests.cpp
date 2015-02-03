#include <iostream>
#include <iomanip>
#include "data_structures.h"
#include "states.h"
void test_build_graph()
{
	unsigned int L = 3;
	data_structures<double> ds;
	build_graph(L, ds);
	std::cout << ds.face_edges << "\n";
	std::cout << ds.adjacent_faces << "\n";
}

void test_homogeneous_state()
{
	unsigned int L = 10;
	double dmu = 0.5, t1 = 1., t2 = 0.3, t3 = 0.1, t4 = 0.05;
	data_structures<arma::cx_double> ds;
	build_graph(L, ds);
	homogeneous_state(dmu, t1, t2, t3, t4, ds);
	ds.psi.save("psi.bin"); 
	ds.w.save("w.bin");
}