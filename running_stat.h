#ifndef __running_stat_h__
#define __running_stat_h__
#include <stdexcept>

template<class type>
class running_stat
{
public:
	running_stat()
	{
		reset();
	}
	void reset()
	{
		c = 0;
		m = s2 = 0.;
	}
	void operator()(const type& x)
	{
		type dm;
		double dc, dc1;
		
		dc1 = c;
		c++;
		dc = c;
		
		dm = (x - m) / dc;
		m += dm;
		s2 += dc1 * dm * dm - s2 / dc;
	}
	void operator()(const running_stat& rs)
	{
		type dm;
		double dc1, dc2, dc12;
		
		dc1 = c;
		dc2 = rs.count();
		dc12 = c + rs.count();
		dc12 /= dc2;
		
		c += rs.count();
		
		dm = (rs.mean() - m) / dc12;
		m += dm;
		s2 += (rs.second_moment() - s2) / dc12 + dm * dm * dc1 / dc2;
	}
	unsigned int count() const
	{
		return c;
	}
	type mean() const
	{
		return m;
	}
	type second_moment() const
	{
		return s2;
	}
	type variance() const
	{
		if(c < 2)
			throw std::logic_error("Estimation of the variance requires at least two values.");
		return s2 * c / (c - 1);
	}
	type variance_of_the_mean() const
	{
		if(c < 2)
			throw std::logic_error("Estimation of the variance requires at least two values.");
		return s2/ (c - 1);
	}

protected:
	unsigned int c;
	type m, s2;
};

#endif
