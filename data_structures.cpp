#include "data_structures.h"
template<class type>
void build_graph(data_structures<type> &ds)
{
	unsigned int L, x, y, i1, i2, i3, i4, i5;
	L = ds.L;
	ds.n_faces = L * L;
	ds.particles.zeros(2 * L * L);
	ds.face_edges.set_size(4, L * L);
	ds.adjacent_faces.set_size(4, L * L);
	for(x = 0; x < L; x++)
	for(y = 0; y < L; y++)
	{
		i1 = x + L * y;
		i2 = (x + 1) % L + L * y;
		i3 = x + L * ((y + 1) % L);
		i4 = (x + L - 1) % L + L * y;
		i5 = x + L * ((y + L - 1) % L);
		ds.face_edges(0, i1) = 2 * i1;
		ds.face_edges(1, i1) = 2 * i2 + 1;
		ds.face_edges(2, i1) = 2 * i3;
		ds.face_edges(3, i1) = 2 * i1 + 1;
		ds.adjacent_faces(0, i1) = i2;
		ds.adjacent_faces(1, i1) = i3;
		ds.adjacent_faces(2, i1) = i4;
		ds.adjacent_faces(3, i1) = i5;
	}
}

template
void build_graph<double>(data_structures<double> &ds);
template
void build_graph<arma::cx_double>(data_structures<arma::cx_double> &ds);




