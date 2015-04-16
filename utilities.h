#ifndef __utilities_h__
#define __utilities_h__

#include <sstream>
#include <ctime>
#include <iomanip>

template <class type>
inline void swap(type &a, type &b)
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

template<class type>
std::string to_string(type x)
{
   static std::stringstream ss;
   ss.str("");
   ss << x;
   return ss.str();
}

inline std::string time_string(double sec)
{
	unsigned int buf;
	static std::stringstream ss;
	
	ss.str("");
	buf = round(fabs(sec));
	
	ss << std::setfill('0');
	ss << std::setw(3)  << buf / 86400 << "-";
	buf %= 86400;
	ss << std::setw(2) << buf / 3600 << ":";
	buf %= 3600;
	ss << std::setw(2) << buf / 60 << ":";
	buf %= 60;
	ss << std::setw(2) << buf;
	
	return ss.str();
}

#ifdef __main_cpp__
	time_t start_time;
#else
	extern time_t start_time;
#endif
	
inline std::string elapsed_time_string()
{
	return time_string(difftime(time(0x0), start_time));
}
#endif

