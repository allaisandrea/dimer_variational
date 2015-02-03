#include <iostream>
#include <iomanip>
#include "data_structures.h"

void test_build_graph()
{
	unsigned int L = 3;
	data_structures<double> ds;
	build_graph(L, ds);
	std::cout << ds.face_edges << "\n";
	std::cout << ds.adjacent_faces << "\n";
}