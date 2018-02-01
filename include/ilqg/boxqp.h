#ifndef BOXQP_H
#define BOXQP_H


#include <Eigen/Dense>
#include <cstddef>

using Eigen::Matrix;
using namespace std;

template<typename type, size_t N, size_t bound>
class BoxQP
{
public:
  typedef Matrix<type, N, N> BoxQPMatrixNd ;
  typedef Matrix<type, N, 1> BoxQPVectorNd ;
  typedef Matrix<type, 1, 1> BoxQPScalar;


  static BoxQPVectorNd minMaxVector(const BoxQPVectorNd &upper,
                                    const BoxQPVectorNd &lower,
                                    const BoxQPVectorNd &y)
  {
    BoxQPVectorNd c_res;

    for (int i = 0; i < N; i++)
    {
      c_res(i) = max(lower(i), min(upper(i), y(i)));
    }

    return c_res;
  }
  static Matrix<type, -1, -1> HFreeMatrix(const BoxQPMatrixNd& H,
                                          const BoxQPVectorNd& free)
  {

    int sum = static_cast<int>(free.sum());

    Matrix<type, -1, -1> H_res(sum, sum);
    Matrix<type, -1, N> H_temp1(sum, N);

    int j = 0;
    int n = (int)H.cols();
    int m = (int)H_temp1.rows();

    for (int i = 0; i < n; i++)
    {
      if (free(i)) {
        H_temp1.row(j) = H.row(i);
        j++;
      }
    }
    j = 0;
    for (int i = 0; i < n; i++)
    {
      if (free(i)) {
        H_res.col(j) = H_temp1.col(i);
        j++;
      }
    }

    return H_res;
  }
  static Matrix<type, -1, 1> grad_free(const BoxQPVectorNd &g,
                                       const BoxQPVectorNd &free)
  {
    int sum = static_cast<int>(free.sum());

    Matrix<type, -1, 1> g_res(sum);

    int j = 0;
    int n = (int)g.size();
    for (int i = 0; i < n; i++)
      if (free(i) == true) {
        g_res(j) = g(i);
        j++;
      }

    return g_res;
  }
  static Matrix<type, -1, 1> search_free(const Matrix<type, -1, -1> &H_free,
                                         const BoxQPVectorNd &grad,
                                         const BoxQPVectorNd &x,
                                         const BoxQPVectorNd &free)
  {
    int n = (int)x.size();
    int sum = static_cast<int>(free.sum());
    BoxQPVectorNd search;
    Matrix<type, -1, 1> x_free(sum), grad_free(sum), search_temp;
    search.setZero();

    int j = 0;
    for (int i = 0; i < n; i++)
      if (free(i)) {
        x_free(j) = x(i);
        j++;
      }

    j = 0;
    for (int i = 0; i < n; i++)
      if (free(i)) {
        grad_free(j) = grad(i);
        j++;
      }

    search_temp = -1.0 * H_free.inverse() *
        ((H_free.transpose()).inverse()*grad_free) - x_free;
    j = 0;
    for (int i = 0; i< n; i++)
      if (free(i)) {
        search(i) = search_temp(j);
        j++;
      }

    return search;
  }

  static void boxQP(const BoxQPMatrixNd &H, const BoxQPVectorNd &g,
                    const BoxQPVectorNd &lower, const BoxQPVectorNd &upper,
                    const BoxQPVectorNd &x0, BoxQPVectorNd& x,
                    int& result, Matrix<type, -1, -1,0,bound,bound>& Hfree, BoxQPVectorNd& free)
  {
    //inputs:
    //   H - positive definite matrix(n * n)
    //g - bias vector(n)
    //lower - lower bounds(n)
    //upper - upper bounds(n)
    //outputs :
    //    x - solution(n)
    //result - result type(roughly, higher is better, see below)
    //Hfree - subspace cholesky factor(n_free * n_free)
    //free - set of free dimensions(n)

    //////// Optimization Option ////

    size_t n = g.size();
    int maxIter = 100;
    double minGrad = 1e-8;
    double minRelImprove = 1e-8;
    double stepDec = 0.6;
    double	minStep = 1e-22;
    double	Armijo = 0.1;
    double sdotg = 0.0;
    bool factorize = false;

    BoxQPVectorNd clamped, old_clamped;
    clamped.setZero();
    free.setOnes();

    double oldvalue = 0.0, gnorm = 0.0, nfactor = 0.0;
    int chol_info, iter;

    result = 0;

    Hfree.setZero();

    /// clamped ///
    x = minMaxVector(upper,lower,x0);

    /// Start Box-QP ///
    BoxQPScalar value;
    BoxQPVectorNd grad;

    value = x.transpose() * g + 0.5 * x.transpose() * H * x;

    for (iter = 1; iter < maxIter; iter++) {
      if (result != 0)
        break;

      if (iter > 1 && (oldvalue - value(0)) < minRelImprove*abs(oldvalue)) {
        result = 4;
        break;
      }
      oldvalue = value(0);
      old_clamped = clamped;
      clamped.setZero();

      grad = g + H*x;

      for (int i = 0; i < n; i++)
      {
        if (x(i) == upper(i) && grad(i) < 0.0)
          clamped(i) = true;
        if (x(i) == lower(i) && grad(i) > 0.0)
          clamped(i) = true;

        free(i) = 1 - clamped(i);
      }

      if (clamped.sum() == n) {
        result = 6;
        break;
      }

      if (iter == 1)
        factorize = true;
      else
      {
        if (old_clamped != clamped)
          factorize = true;
        else
          factorize = false;
      }

      /// for eq. (15)
      if (factorize)
      {
        Eigen::LLT< Matrix<type, -1, -1> > chol(HFreeMatrix(H, free));
        Hfree = chol.matrixL().transpose();
        chol_info = chol.info();

        //Hfree = R;
        if (chol_info)
        {
          result = -1;
          break;
        }
        nfactor = nfactor + 1;
      }

      gnorm = grad_free(grad, free).norm();
      if (gnorm < minGrad) {
        result = 5;
        break;
      }

      BoxQPVectorNd grad_c;
      BoxQPVectorNd x_c, search;
      double step;
      int nstep;
      BoxQPVectorNd xc;
      BoxQPScalar vc;

      for (int i = 0; i < n; i++)
        x_c(i) = clamped(i) * x(i);


      grad_c = g + H * x_c;
      search.setZero();
      search = search_free(Hfree, grad_c, x, free);

      sdotg = search.transpose() * grad;
      if (sdotg >= 0)
        break;

      step = 1;
      nstep = 0;

      xc = minMaxVector(upper, lower, x + step*search);
      vc = xc.transpose() * g + 0.5*xc.transpose() * H * xc;

      while ((vc(0) - oldvalue) / (step*sdotg) < Armijo) {
        step = step*stepDec;
        nstep = nstep + 1;
        xc = minMaxVector(upper, lower, x + step*search);
        vc = xc.transpose() * g + 0.5*xc.transpose() * H * xc;
        if (step < minStep) {
          result = 2;
          break;
        }
      }

      x = xc;
      value = vc;

    }//iteration

    if (iter >= maxIter)
      result = 1;

    /*switch (result)
            {
            case -1:
            cout << "Hessian is not positive" << endl;
            break;
            case 0:
            cout << "No descent direction found" << endl;
            break;
            case 1:
            cout << "Maximum main iterations exceeded" << endl;
            break;
            case 2:
            cout << "Maximum line-search iterations exceeded" << endl;
            break;
            case 3:
            cout << "No bounds, returning Newton point" << endl;
            break;
            case 4:
            cout << "Improvement smaller than tolerance" << endl;
            break;
            case 5:
            cout << "Gradient norm smaller than tolerance" << endl;
            break;
            case 6:
            cout << "All dimensions are clamped" << endl;
            break;
            default:
            break;
            }*/

  }
};

#endif
