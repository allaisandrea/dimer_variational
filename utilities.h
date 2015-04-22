#ifndef __utilities_h__
#define __utilities_h__

#include <sstream>
#include <ctime>
#include <iomanip>
#include <mpi.h>

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

inline void MPI_Send(arma::vec& x, int dest, int tag, MPI_Comm comm)
{
	int err;
	arma::uword n_elem = x.n_elem;
	err = MPI_Send(&n_elem, sizeof(n_elem), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS)
		throw std::runtime_error("MPI_Send failed");
	err = MPI_Send(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, dest, tag, comm);
	if(err != MPI_SUCCESS)
		throw std::runtime_error("MPI_Send failed");
}

inline void MPI_Send(const arma::vec& _x, int dest, int tag, MPI_Comm comm)
{
	arma::vec x(_x);
	MPI_Send(x, dest, tag, comm);
}

inline void MPI_Recv(arma::vec& x, int source, int tag, MPI_Comm comm)
{
	int err;
	arma::uword n_elem;
	err = MPI_Recv(&n_elem, sizeof(n_elem), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS)
		throw std::runtime_error("MPI_Recv failed");
	x.set_size(n_elem);
	err = MPI_Recv(x.memptr(), x.n_elem * sizeof(double), MPI_BYTE, source, tag, comm, MPI_STATUS_IGNORE);
	if(err != MPI_SUCCESS)
		throw std::runtime_error("MPI_Recv failed");
}

#endif

