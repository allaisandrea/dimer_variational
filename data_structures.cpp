#include "data_structures.h"
template<class type>
void build_graph(unsigned int L, data_structures<type> &ds)
{
	unsigned int x, y, i1, i2, i3, i4, i5;
	ds.L = L;
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
		face_edges(0, i1) = 2 * i1;
		face_edges(1, i1) = 2 * i2 + 1;
		face_edges(2, i1) = 2 * i3;
		face_edges(3, i1) = 2 * i1 + 1;
		adjacent_faces(0, i1) = i2;
		adjacent_faces(1, i1) = i3;
		adjacent_faces(2, i1) = i4;
		adjacent_faces(3, i1) = i5;
	}
}





