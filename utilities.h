#ifndef __utilities_h__
#define __utilities_h__

template <class type>
void swap(type &a, type &b)
{
	type c;
	c = a;
	a = b;
	b = c;
}

inline double abs_squared(double x)
{
	return x * x;
}

inline double abs_squared(arma::cx_double x){
	return norm(x);
}

#endif

