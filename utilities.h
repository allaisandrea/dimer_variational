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

inline double real(double x)
{
	return x;
}

inline double real(arma::cx_double x)
{
	return x.real();
}

inline double conj(double x)
{
	return x;
}

inline arma::cx_double conj(arma::cx_double x)
{
	return conj(x);
}


inline arma::uvec set_to_uvec(const std::set<unsigned int> &s)
{
	unsigned int c;
	std::set<unsigned int>::iterator i;
	arma::uvec v;
	v.set_size(s.size());
	
	c = 0;
	for(i = s.begin(); i != s.end(); ++i)
		v(++c) = *i;
	return v;
}
#endif

