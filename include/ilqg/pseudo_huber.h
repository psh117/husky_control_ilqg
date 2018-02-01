#ifndef PSEUDO_HUBER_H
#define PSEUDO_HUBER_H

#include <Eigen/Dense>

template<typename vector_type>
vector_type sabs(const vector_type& x, const vector_type& p) // pseudo-Huber
{
  size_t m = x.size();
  vector_type y;

  for (size_t i = 0; i < m; i++)
    y(i) = sqrt(pow(x(i), 2) + pow(p(i), 2)) - p(i);

  return y;
}

#endif
