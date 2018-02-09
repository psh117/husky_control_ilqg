/**
  @file ilqg.h
  @brief iLQG C++ Implementation
  @author Sanghyun Kim (ggory15@snu.ac.kr), Suhan Park (psh117@snu.ac.kr)

  @details
  Original Paper

 BIBTeX:

 @INPROCEEDINGS{
 author={Tassa, Y. and Mansard, N. and Todorov, E.},
 booktitle={Robotics and Automation (ICRA), 2014 IEEE International Conference on},
 title={Control-Limited Differential Dynamic Programming},
 year={2014}, month={May}, doi={10.1109/ICRA.2014.6907001}}

  */

#ifndef ILQG_H
#define ILQG_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <omp.h>
#include "boxqp.h"

using namespace Eigen;
using namespace std;

/**
 * @brief The iLQGParameter struct
 */
struct iLQGParameter
{
  iLQGParameter() ///< Default values
  {
    alpha = ArrayXd::LinSpaced(11, 0, -3);
    for(int i=0; i<alpha.size(); i++)
    {
      alpha(i) = pow(10, alpha(i));
    }
    parallel_search = true;
    tol_fun = 1e-7;
    tol_grad = 1e-4;
    max_iter = 500;
    lambda = 1.;
    dlambda = 1.;
    lambda_factor = 1.6;
    lambda_max = 1e10;
    lambda_min = 1e-6;
    reg_type = RegByQuu;
    z_min = 0.;
  }

  ArrayXd alpha;
  bool parallel_search; ///< Use parallel line-search?
  double tol_fun;
  double tol_grad;
  size_t max_iter;
  double lambda;
  double dlambda;
  double lambda_factor;
  double lambda_max;
  double lambda_min;

  enum RegularizationType {RegByQuu, RegByVxx};
  RegularizationType reg_type;
  double z_min;

};

/**
 * @brief The iLQG class
 * @todo USE constexpr to make it real-time safe or more faster.
 */

// State(X), Input(U) - fixed, Time-horizon(N) - variable
template<typename data_type, size_t x_dim, size_t u_dim>
class iLQG
{
  // Vector<N> = Matrix<data_type, N, 1>
protected:
  template<size_t dim>
  using Vector = Matrix<data_type, dim, 1>;

  typedef Vector<x_dim> VectorX;
  typedef Vector<u_dim> VectorU;
  //typedef Vector<N_dim> VectorN;
  typedef Vector<x_dim + u_dim> VectorXU;
  typedef Matrix<data_type, x_dim, x_dim> MatrixXX;
  typedef Matrix<data_type, x_dim, u_dim> MatrixXU;
  typedef Matrix<data_type, u_dim, x_dim> MatrixUX;
  typedef Matrix<data_type, u_dim, u_dim> MatrixUU;
  typedef Matrix<data_type, Dynamic, Dynamic, 0, x_dim+u_dim, x_dim+u_dim> Matrixxx;

protected:
  iLQG();

public:
  void init(const VectorX& initial_state, ///< x0
            const MatrixXd& initial_input,  ///< u0
            size_t time_horizon);     ///< Time horizon length

  void init(const VectorX &initial_state, ///< x0
            const MatrixXd &initial_input,  ///< u0
            size_t time_horizon,      ///< Time horizon length
            const iLQGParameter &param);    ///< User Parameter (Optional)

  void setDesiredState(const VectorX& state)  {xd_ = state;}
  void setInputConstraint(MatrixXd &u_limit);
  void plan();
  void replan(size_t tick, const VectorX &state);
  void updateCurrentState(const VectorX& state);

  const MatrixXd& getPlannedStateMatrix() {return x_;}
  const MatrixXd& getPlannedControlInputMatrix() {return u_;}
  const VectorXd& getPlannedCostVector() {return c_;}
  const size_t getHorizon() const { return N_; }

  // Verbose Levels
  enum verbose_level {vbl_no_print, vbl_warning, vbl_info, vbl_verbose, vbl_debug};
  void setVerboseLevel(verbose_level level) { verbose_ = level; }

private:
  void initDimension();

  void forwardPassInit(const double alpha, MatrixXd & x_new, MatrixXd & u_new, VectorXd & c_new);
  void forwardPassInit(const double alpha, MatrixXd & x_new, MatrixXd & u_new, VectorXd & c_new, size_t tick);

  void backwardPass(const double lambda,
                    int &diverge, VectorX& Vx, MatrixXX& Vxx,  Vector2d& dV);
  void backwardPass(const double lambda,
                    int &diverge, VectorX& Vx, MatrixXX& Vxx,  Vector2d& dV, size_t tick);
  void forwardPass(const double alpha,
                   MatrixXd &x_new, MatrixXd &u_new, VectorXd &c_new);
  void forwardPass(const double alpha,
                   MatrixXd &x_new, MatrixXd &u_new, VectorXd &c_new, size_t tick);

  void dynamicsCost(const VectorX &x, const VectorU& u, const size_t index, MatrixXX &fx, MatrixXU &fu,
                    VectorX& cx, VectorU& cu, MatrixXX& cxx, MatrixXU& cxu, MatrixUU& cuu);
  void QuxFreeMatrix(const MatrixUX &Qux, const VectorU &free, Matrixxx &Qux_free);
  double absMaxVector(const VectorU &x, const VectorU &u);
  //VIRTUAL
  virtual const VectorX dynamicsCalc(const VectorX& x, const VectorU& u) = 0;
  virtual double costFunction(const VectorX& x, const VectorU& u, size_t index) = 0;

protected:
  VectorX xd_;        ///< Desired state vector (n_)
private:
  const size_t n_;		///< Dimension of the state vector
  const size_t m_;		///< Dimension of the control input vector
  size_t k_;          ///< Dimension of the backtracking alpha vector
  size_t N_;          ///< Time horizon length

  MatrixXd x_;        ///< State vector (n_, N_)
  MatrixXd u_;        ///< Control input vector (m_, N_)
  VectorXd c_;        ///< Cost vector (N_)

  VectorX x0_;        ///< Initial state vector (n_)
  MatrixXd u0_;       ///< Initial control input Matrix (m_, N_)

  MatrixXd u_limit_;  ///< (i, 0) is lower bound, (i, 1) is upper bound
  bool use_u_limit_;  ///< Whether iLQG uses control input limit

  vector<MatrixXd> x_new_;
  vector<MatrixXd> u_new_;
  vector<VectorXd> c_new_;

  vector<MatrixXX> fx_;
  vector<MatrixXU> fu_;

  vector<MatrixXX> cxx_;
  vector<MatrixXU> cxu_;
  vector<MatrixUU> cuu_;

  vector<VectorX> cx_;
  vector<VectorU> cu_;

  vector<MatrixUX> L_;
  vector<VectorU> l_;

  VectorXd D_cost_;   ///< Costs wrt alpha (k_)

  bool diverge_;

  iLQGParameter params_;

  verbose_level verbose_;
};

template<typename data_type, size_t x_dim, size_t u_dim>
iLQG<data_type, x_dim, u_dim>::iLQG() : use_u_limit_(false), n_(x_dim), m_(u_dim), verbose_(vbl_info)
{
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::init(const VectorX &initial_state,
                                         const MatrixXd &initial_input,
                                         size_t time_horizon)
{
  if (initial_state.rows() != n_)
  {
    // ERROR
    if(verbose_ >= vbl_warning)
    {
      std::cout << "STATE DIMENSION ERROR!!!" << endl;
    }
  }
  if (initial_input.rows() != m_)
  {
    // ERROR
    if(verbose_ >= vbl_warning)
    {
      std::cout << "INPUT DIMENSION ERROR!!!" << endl;
    }
  }

  x0_ = initial_state;
  u0_ = initial_input;

  N_ = time_horizon;

  initDimension();
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::init(const VectorX & initial_state,
                                         const MatrixXd& initial_input,
                                         size_t time_horizon,
                                         const iLQGParameter& param)
{
  params_ = param;    // Update user parameters
  init(initial_state, initial_input, time_horizon);
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::setInputConstraint(MatrixXd &u_limit)
{
  u_limit_ = u_limit;
  use_u_limit_ = true;
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::initDimension()
{
  k_ = params_.alpha.size();

  x_.resize(n_, N_);
  u_.resize(m_, N_);
  c_.resize(N_);

  x_new_.resize(k_);
  u_new_.resize(k_);
  c_new_.resize(k_);

  D_cost_.resize(k_);


  fx_.resize(N_);
  fu_.resize(N_);

  cxx_.resize(N_);
  cxu_.resize(N_);
  cuu_.resize(N_);

  cx_.resize(N_);
  cu_.resize(N_);

  L_.resize(N_-1);
  l_.resize(N_);

  for(size_t i=0; i<k_; i++)
  {
    x_new_[i].resize(n_, N_);
    u_new_[i].resize(m_, N_);
    c_new_[i].resize(N_);
  }
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::
forwardPassInit(const double alpha, MatrixXd & x_new,
                MatrixXd & u_new, VectorXd & c_new)
{
  x_new.col(0) = x0_;
  u_new.setZero();

  // TODO: (non-realtime)
  u_new = alpha * u0_;

  // Naive clamping
  if (use_u_limit_)
  {
    for (size_t i=0; i<N_-1; i++)
    {
      for(size_t j=0; j<m_; j++)
      {
        u_new(j, i) = std::max(u_limit_(j, 0),
                               std::min(u_limit_(j, 1),
                                        u_new(j, i)));
      }
    }
    for(size_t j=0; j<m_; j++)
    {
      u_new(j, N_-1) = 0.;
    }
  }

  for (size_t i=0; i<N_-1; i++)
  {
    x_new.col(i+1) = dynamicsCalc(x_new.col(i), u_new.col(i)); // Eq (1)
    c_new(i) = costFunction(x_new.col(i), u_new.col(i), i);
  }
  c_new(N_-1) = costFunction(x_new.col(N_-1), u_new.col(N_-1), N_-1);
}
template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::
forwardPassInit(const double alpha, MatrixXd & x_new,
                MatrixXd & u_new, VectorXd & c_new, size_t tick)
{
  x_new.col(0) = x0_;
  u_new.setZero();

  // TODO: (non-realtime)
  u_new = alpha * u0_;

  // Naive clamping
  if (use_u_limit_)
  {
    for (size_t i=tick; i<N_-1; i++)
    {
      for(size_t j=0; j<m_; j++)
      {
        u_new(j, i) = std::max(u_limit_(j, 0),
                               std::min(u_limit_(j, 1),
                                        u_new(j, i)));
      }
    }
    for(size_t j=0; j<m_; j++)
    {
      u_new(j, N_-1) = 0.;
    }
  }

  for (size_t i=tick; i<N_-1; i++)
  {
    x_new.col(i+1) = dynamicsCalc(x_new.col(i), u_new.col(i)); // Eq (1)
    c_new(i) = costFunction(x_new.col(i), u_new.col(i), i);
  }
  c_new(N_-1) = costFunction(x_new.col(N_-1), u_new.col(N_-1), N_-1);
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::backwardPass(const double lambda,
                                                 int &diverge, VectorX& Vx,MatrixXX& Vxx, Vector2d& dV)
{
  MatrixXX Vxx_reg, Qxx;
  VectorX Qx;
  VectorU Qu;
  MatrixUU Quu, QuuF;
  MatrixUX Qux, Qux_reg;

  Matrixxx R;
  Matrixxx L_free;
  Matrixxx Qux_free;
  VectorU free;

  dV.setZero();

  diverge = 0;
  Vx = cx_[N_ -1];
  Vxx = cxx_[N_ -1];

  for (long i = N_ -2; i>=0; i--)
  {
    Qx = cx_[i] + fx_[i].transpose() * Vx;
    Qu = cu_[i] + fu_[i].transpose() * Vx;
    Qxx = cxx_[i] + fx_[i].transpose() * Vxx * fx_[i];
    Qux = cxu_[i].transpose() + fu_[i].transpose() * Vxx * fx_[i];
    Quu = cuu_[i] + fu_[i].transpose() * Vxx * fu_[i];

    if (params_.reg_type == iLQGParameter::RegByVxx)
    {
      Vxx_reg = Vxx + lambda * MatrixXX::Identity();
    }
    else
    {
      Vxx_reg = Vxx;
    }

    Qux_reg = cxu_[i].transpose() + fu_[i].transpose() * Vxx_reg * fx_[i];

    if(params_.reg_type == iLQGParameter::RegByQuu)
    {
      QuuF = cuu_[i] + fu_[i].transpose() * Vxx_reg * fu_[i]
          + lambda * MatrixUU::Identity();
    }
    else
    {
      QuuF = cuu_[i] + fu_[i].transpose() * Vxx_reg * fu_[i];
    }

    if (use_u_limit_)
    {
      VectorU lower, upper;
      lower = u_limit_.col(0) - u_.col(i);
      upper = u_limit_.col(1) - u_.col(i);

      l_[i].setZero();
      L_[i].setZero();

      int QP_res;
      BoxQP<data_type, u_dim, x_dim + u_dim>::boxQP(QuuF, Qu, lower, upper,
                                     l_[std::min<long>(i + 1, N_ - 1)], l_[i], QP_res, R, free);

      if (QP_res < 1)
      {
        diverge = i;
        // return from backpass();
      }

      size_t j=0;
      if(free.sum() > 0.)
      {
        QuxFreeMatrix(Qux, free, Qux_free);
        L_free = -1.0*R.inverse()*((R.transpose()).inverse())* Qux_free;

        for (size_t k = 0; k < m_; k++)
        {
          if (free(k))
          {
            L_[i].row(k) = L_free.row(j);
            j++;
          }
        }
      }

    }
    else // !use_u_limit_
    {
      LLT<MatrixUU> chol(QuuF);
      R = chol.matrixL().transpose();
      if(chol.info())
      {
        diverge = i;

        if(verbose_ >= vbl_warning)
        {
          cout << "Warning: Failed Cholosky decomposition" << endl;
        }
      }

      Matrix<data_type, u_dim, x_dim + 1> KK;
      Matrix<data_type, u_dim, x_dim + 1> Qs;
      Qs.col(0) = Qu;
      Qs.template topRightCorner<u_dim, x_dim>() = Qux_reg;

      KK = -1.0*R.inverse()* (R.transpose().inverse() * Qs);

      l_[i] = KK.template block<u_dim, 1>(0, 0);
      L_[i] = KK.template block<u_dim, x_dim>(0, 1);
    }

    Vector2d ddV;
    ddV(0) = l_[i].transpose() * Qu;
    ddV(1) = 0.5 * l_[i].transpose() * Quu * l_[i];   // 0.5 is added, sign??
    dV = dV + ddV;
    Vx = Qx + L_[i].transpose()*Quu*l_[i] + L_[i].transpose() * Qu + Qux.transpose() * l_[i];
    Vxx = Qxx + L_[i].transpose() * Quu *L_[i] + L_[i].transpose() *Qux + Qux.transpose()*L_[i];
    Vxx = 0.5* (Vxx + Vxx.transpose());
  }
}


template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::backwardPass(const double lambda,
                                                 int &diverge, VectorX& Vx,MatrixXX& Vxx, Vector2d& dV,
                                                 size_t tick)
{
  MatrixXX Vxx_reg, Qxx;
  VectorX Qx;
  VectorU Qu;
  MatrixUU Quu, QuuF;
  MatrixUX Qux, Qux_reg;

  Matrixxx R;
  Matrixxx L_free;
  Matrixxx Qux_free;
  VectorU free;

  dV.setZero();

  diverge = 0;
  Vx = cx_[N_ -1];
  Vxx = cxx_[N_ -1];

  for (long i = N_ -2; i>=tick; i--)
  {
    Qx = cx_[i] + fx_[i].transpose() * Vx;
    Qu = cu_[i] + fu_[i].transpose() * Vx;
    Qxx = cxx_[i] + fx_[i].transpose() * Vxx * fx_[i];
    Qux = cxu_[i].transpose() + fu_[i].transpose() * Vxx * fx_[i];
    Quu = cuu_[i] + fu_[i].transpose() * Vxx * fu_[i];

    if (params_.reg_type == iLQGParameter::RegByVxx)
    {
      Vxx_reg = Vxx + lambda * MatrixXX::Identity();
    }
    else
    {
      Vxx_reg = Vxx;
    }

    Qux_reg = cxu_[i].transpose() + fu_[i].transpose() * Vxx_reg * fx_[i];

    if(params_.reg_type == iLQGParameter::RegByQuu)
    {
      QuuF = cuu_[i] + fu_[i].transpose() * Vxx_reg * fu_[i]
          + lambda * MatrixUU::Identity();
    }
    else
    {
      QuuF = cuu_[i] + fu_[i].transpose() * Vxx_reg * fu_[i];
    }

    if (use_u_limit_)
    {
      VectorU lower, upper;
      lower = u_limit_.col(0) - u_.col(i);
      upper = u_limit_.col(1) - u_.col(i);

      l_[i].setZero();
      L_[i].setZero();

      int QP_res;
      BoxQP<data_type, u_dim, x_dim + u_dim>::boxQP(QuuF, Qu, lower, upper,
                                     l_[std::min<long>(i + 1, N_ - 1)], l_[i], QP_res, R, free);

      if (QP_res < 1)
      {
        diverge = i;
        // return from backpass();
      }

      size_t j=0;
      if(free.sum() > 0.)
      {
        QuxFreeMatrix(Qux, free, Qux_free);
        L_free = -1.0*R.inverse()*((R.transpose()).inverse())* Qux_free;

        for (size_t k = 0; k < m_; k++)
        {
          if (free(k))
          {
            L_[i].row(k) = L_free.row(j);
            j++;
          }
        }
      }

    }
    else // !use_u_limit_
    {
      LLT<MatrixUU> chol(QuuF);
      R = chol.matrixL().transpose();
      if(chol.info())
      {
        diverge = i;

        if(verbose_ >= vbl_warning)
        {
          cout << "Warning: Failed Cholosky decomposition" << endl;
        }
      }

      Matrix<data_type, u_dim, x_dim + 1> KK;
      Matrix<data_type, u_dim, x_dim + 1> Qs;
      Qs.col(0) = Qu;
      Qs.template topRightCorner<u_dim, x_dim>() = Qux_reg;

      KK = -1.0*R.inverse()* (R.transpose().inverse() * Qs);

      l_[i] = KK.template block<u_dim, 1>(0, 0);
      L_[i] = KK.template block<u_dim, x_dim>(0, 1);
    }

    Vector2d ddV;
    ddV(0) = l_[i].transpose() * Qu;
    ddV(1) = 0.5 * l_[i].transpose() * Quu * l_[i];   // 0.5 is added, sign??
    dV = dV + ddV;
    Vx = Qx + L_[i].transpose()*Quu*l_[i] + L_[i].transpose() * Qu + Qux.transpose() * l_[i];
    Vxx = Qxx + L_[i].transpose() * Quu *L_[i] + L_[i].transpose() *Qux + Qux.transpose()*L_[i];
    Vxx = 0.5* (Vxx + Vxx.transpose());
  }
}



template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::forwardPass(const double alpha,
                                                MatrixXd& x_new,
                                                MatrixXd& u_new,
                                                VectorXd& c_new)
{
  x_new.col(0) =  x0_;
  u_new.setZero();

  for (size_t j = 0; j < N_ - 1; j++) {
    u_new.col(j) = u_.col(j) + alpha *l_[j];
    u_new.col(j) = u_new.col(j) + L_[j] *
        (x_new.col(j) - x_.col(j));

    if (use_u_limit_) {
      for (size_t i = 0; i < m_; i++) {
        u_new.col(j)(i) = max(u_limit_(i, 0),
                              min(u_limit_(i, 1),
                                  u_new.col(j)(i)));
      }
    }

    x_new.col(j + 1) = dynamicsCalc(x_new.col(j), u_new.col(j));
    c_new(j) = costFunction(x_new.col(j), u_new.col(j), j);
  }

  c_new(N_ - 1) = costFunction(x_new.col(N_ - 1), VectorU::Zero(), N_ - 1);
}


template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::forwardPass(const double alpha,
                                                MatrixXd& x_new,
                                                MatrixXd& u_new,
                                                VectorXd& c_new,
                                                size_t tick)
{
  x_new.col(0) =  x0_;
  u_new.setZero();

  for (size_t j = tick; j < N_ - 1; j++) {
    u_new.col(j) = u_.col(j) + alpha *l_[j];
    u_new.col(j) = u_new.col(j) + L_[j] *
        (x_new.col(j) - x_.col(j));

    if (use_u_limit_) {
      for (size_t i = 0; i < m_; i++) {
        u_new.col(j)(i) = max(u_limit_(i, 0),
                              min(u_limit_(i, 1),
                                  u_new.col(j)(i)));
      }
    }

    x_new.col(j + 1) = dynamicsCalc(x_new.col(j), u_new.col(j));
    c_new(j) = costFunction(x_new.col(j), u_new.col(j), j);
  }

  c_new(N_ - 1) = costFunction(x_new.col(N_ - 1), VectorU::Zero(), N_ - 1);
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::
dynamicsCost(const VectorX &x, const VectorU& u,const size_t index, MatrixXX &fx, MatrixXU &fu,
             VectorX& cx, VectorU& cu, MatrixXX& cxx, MatrixXU& cxu, MatrixUU& cuu)
{

  // TODO: Organizing local variables
  const double h = pow(2.0, -17);
  {
    // Dynamics
    Matrix<data_type, x_dim + u_dim, x_dim + u_dim> H =
        h *
        Matrix<data_type, x_dim + u_dim, x_dim + u_dim>
        ::Identity();

    for (size_t i=0; i< n_+ m_; i++)
    {
      H.col(i).template head<x_dim>() += x;
      H.col(i).template tail<u_dim>() += u;
    }

    Matrix<data_type, x_dim, x_dim + u_dim + 1> Y;
    Matrix<data_type, x_dim, x_dim + u_dim> J;

    Y.col(0) = dynamicsCalc(x,u);
    for (size_t i=0; i<n_+m_; i++)
    {
      Y.col(i+1) = dynamicsCalc(
            H.col(i).template head<x_dim>(),
            H.col(i).template tail<u_dim>());
    }
    // Finite difference
    for (size_t i=0; i<n_+m_; i++)
    {
      J.col(i) = (Y.col(i+1) - Y.col(0)) / h;
    }
    fx = J.template block<x_dim, x_dim>(0,0);
    fu = J.template block<x_dim, u_dim>(0,x_dim);
  }
  {
    Matrix<data_type, x_dim + u_dim, x_dim + u_dim> H =
        h *
        Matrix<data_type, x_dim + u_dim, x_dim + u_dim>
        ::Identity();
    Vector<x_dim + u_dim+1>  Y;
    VectorXU J;
    for (size_t i=0; i< n_+ m_; i++)
    {
      H.col(i).template head<x_dim>() += x;
      H.col(i).template tail<u_dim>() += u;
    }
    Y(0) = costFunction(x,u,index);
    for (size_t i=0; i<n_+m_; i++)
    {
      Y(i+1) = costFunction(
            H.col(i).template head<x_dim>(),
            H.col(i).template tail<u_dim>(), index);
    }
    // Finite difference
    for (size_t i=0; i<n_+m_; i++)
    {
      J(i) = (Y(i+1) - Y(0)) / h;
    }
    cx = J.template segment<x_dim>(0);
    cu = J.template segment<u_dim>(x_dim);
  }

  // Cost
  {
    VectorXU X;
    X.template head<x_dim>() = x;
    X.template tail<u_dim>() = u;


    Matrix<data_type, x_dim+u_dim, 4> Y;
    Matrix<data_type, x_dim+u_dim, x_dim+u_dim> JJ;
    VectorXU J;
    Y.setZero();
    JJ.setZero();
    J.setZero();

    Y.col(0) = X;

    double fxy, fx, fy, f0;
    f0 = costFunction(Y.col(0).template segment<x_dim>(0),
                      Y.col(0).template segment<u_dim>(x_dim),
                      index);

    for (size_t i=0; i<n_+m_; i++)
    {
      Y.col(1) = X;
      Y.col(1)(i) = Y.col(1)(i) + h;
      for (size_t j=0; j< n_+m_; j++)
      {
        Y.col(2) = X;
        Y.col(3) = Y.col(1);
        Y.col(2)(j) = Y.col(2)(j) + h;
        Y.col(3)(j) = Y.col(3)(j) + h;

        fxy = costFunction(Y.col(3).template segment<x_dim>(0),
                           Y.col(3).template segment<u_dim>(x_dim),
                           index);
        fx = costFunction(Y.col(2).template segment<x_dim>(0),
                          Y.col(2).template segment<u_dim>(x_dim),
                          index);
        fy = costFunction(Y.col(1).template segment<x_dim>(0),
                          Y.col(1).template segment<u_dim>(x_dim),
                          index);

        JJ(i, j) = (fxy - fx - fy + f0) / h / h;
      }
    }
    JJ = (JJ.transpose() + JJ) * 0.5;
    cxx = JJ.template block<x_dim, x_dim>(0,0);
    cxu = JJ.template block<x_dim, u_dim>(0,x_dim);
    cuu = JJ.template block<u_dim, u_dim>(x_dim, x_dim);
  }

}

template<typename data_type, size_t x_dim, size_t u_dim>
inline void iLQG<data_type, x_dim, u_dim>::QuxFreeMatrix(
    const MatrixUX &Qux, const VectorU &free, Matrixxx& Qux_free)
{

  size_t m = Qux.rows();
  size_t n = Qux.cols();
  size_t sum = static_cast<size_t>(free.sum());
  Qux_free.resize(sum, n);
  size_t j = 0;

  for (size_t i = 0; i < m; i++)
  {
    if (free(i)) {
      Qux_free.row(j) = Qux.row(i);
      j++;
    }
  }
}


template<typename data_type, size_t x_dim, size_t u_dim>
inline double iLQG<data_type, x_dim, u_dim>::absMaxVector(const VectorU &x, const VectorU &u)
{
  double res;
  VectorU temp_vec;
  for (int i = 0; i < m_; i++) {
    temp_vec(i) = abs(x(i)) / abs(u(i) + 1.0);
  }

  res = temp_vec.maxCoeff();
  return res;
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::replan(size_t tick, const VectorX &state)
{
cout << "start to plan" << endl;
  x_.col(tick) = state;

  bool is_dynamics_changed_after_tick = true;
  bool is_backpass_done = false;
  bool is_forwardpass_done = false;
  // Control limits

  double lambda, dlambda, gnorm;
  double d_cost, alpha, expected, z;
  lambda = params_.lambda;
  dlambda = params_.dlambda;

  VectorX Vx;
  MatrixXX Vxx;

  size_t w; // proper alpha index

  Vector2d dV;

  // line no. 141 (matlab)
  diverge_ = true;
cout << "start to plan2" << endl;
  for (size_t i=0; i<k_; i++)
  {
    x_new_[i].col(tick) = state;
    forwardPassInit(params_.alpha(i), x_new_[i], u_new_[i], c_new_[i],tick);
    // Simplistic divergence test (1st = solution)
    if (x_new_[i].maxCoeff() < 1e8)
    {
      u_ = u_new_[i];
      x_ = x_new_[i];
      c_ = c_new_[i];
      diverge_ = true;
      break;
    }
  }

  if (diverge_)
  {

  }

  for (size_t iter=1; iter < 2; iter++)
  {
    // ====== STEP 1: differentiate dynamics and cost along new trajectory
    if(is_dynamics_changed_after_tick)
    {
      //#pragma omp parallel for
      for (size_t i = tick; i < N_; i++)
      {
        dynamicsCost(x_.col(i), u_.col(i), i, fx_[i], fu_[i],
                     cx_[i], cu_[i], cxx_[i], cxu_[i], cuu_[i]);

      }
    }

cout << "start to backwardpass" << endl;
    // ====== STEP 2: backward pass, compute optimal control law and cost-to-go
    // Eq. (4) ~ (6)
    is_backpass_done = false;
    int backpass_diverge;
    while (!is_backpass_done)
    {
      backwardPass(lambda,backpass_diverge,Vx,Vxx,dV, tick);
      if (backpass_diverge) {
        dlambda = max(dlambda * params_.lambda_factor, params_.lambda_factor);
        lambda = max(lambda * dlambda, params_.lambda_min);
        if (lambda > params_.lambda_max) {
          if(verbose_ >= vbl_warning)
          {
            cout << "Warning: Lambda caused divergence" << endl;
          }
          break;
        }
        continue;
      }
      is_backpass_done = true;
    }

    gnorm = 0.0;
    for (int i = 0; i < N_ - 1; i++)
      gnorm = gnorm + absMaxVector(l_[i], u_.col(i));

    gnorm = gnorm / (N_ - 1);
    if (gnorm < params_.tol_grad && lambda < 1e-5)
    {
      dlambda = min(dlambda / params_.lambda_factor, 1.0 / params_.lambda_factor);
      if (lambda > params_.lambda_min)
        lambda = lambda * dlambda;

      if(verbose_ >= vbl_warning)
      {
        cout << "Warning:: Break due to small gradient" << endl;
      }
      break;
    }

cout << "start to forwardpass" << endl;
    // ====== STEP 3: line-search to find new control sequence, trajectory, cost
    is_forwardpass_done = false;
    if (is_backpass_done)
    {
      //#pragma omp parallel for
      for (size_t al_i = 0; al_i < k_; al_i++) // alpha
      {
        forwardPass(params_.alpha(al_i),x_new_[al_i], u_new_[al_i], c_new_[al_i], tick);
      }
	cout << "D_cost_" << endl;
      for (int i = 0; i < k_; i++)
        D_cost_(i) = (c_.tail(N_-tick-1)).sum() - (c_new_[i].tail(N_-tick-1)).sum();

      d_cost = D_cost_.maxCoeff(&w);
      alpha = params_.alpha(w);
      expected = -1.0*alpha*(dV(0) + alpha*dV(1));


      if (expected > 0)
        z = d_cost / expected;
      else {
        z = d_cost / abs(d_cost);
        if(verbose_ >= vbl_warning)
        {
          cout << "Warning:: non-positive expected reduction" << endl;
        }
      }
      if (z > params_.z_min)
        is_forwardpass_done = true;
      else
        alpha = 0.0;

    }

    if (is_forwardpass_done)
    {

      if(verbose_ >= vbl_verbose)
      {
        cout << "iteration:" << iter << " Cost " << c_.sum() << " dCost " << d_cost  <<
                " expected " << expected << " gradient " << gnorm <<endl;
      }
      c_ = c_new_[w];
      x_ = x_new_[w];
      u_ = u_new_[w];
      u0_ = u_;

      dlambda = min(dlambda / params_.lambda_factor, 1.0 / params_.lambda_factor);
      if (lambda > params_.lambda_min)
        lambda = lambda * dlambda;

      if (d_cost < params_.tol_fun) {
        if(verbose_ >= vbl_info)
        {
          cout << "Successful convergence" << endl;
        }
        break;
      }

      is_dynamics_changed_after_tick = true;
    }
    else
    {
      if(verbose_ >= vbl_verbose)
      {
        cout << "iteration:" << iter << " " << "NO STEP" << endl;
      }
      dlambda = max(dlambda * params_.lambda_factor, params_.lambda_factor);
      lambda = max(lambda * dlambda, params_.lambda_min);
      if (lambda > params_.lambda_max) {
        if(verbose_ >= vbl_warning)
        {
          cout << "Warning:: non-positive expected reduction" << endl;
        }
        break;
      }
    }

  }
}

template<typename data_type, size_t x_dim, size_t u_dim>
void iLQG<data_type, x_dim, u_dim>::plan()
{
  bool is_dynamics_changed_after_tick = true;
  bool is_backpass_done = false;
  bool is_forwardpass_done = false;
  // Control limits

  double lambda, dlambda, gnorm;
  double d_cost, alpha, expected, z;
  lambda = params_.lambda;
  dlambda = params_.dlambda;

  VectorX Vx;
  MatrixXX Vxx;

  size_t w; // proper alpha index

  Vector2d dV;

  // line no. 141 (matlab)
  diverge_ = true;
  for (size_t i=0; i<k_; i++)
  {
    forwardPassInit(params_.alpha(i), x_new_[i], u_new_[i], c_new_[i]);
    // Simplistic divergence test (1st = solution)
    if (x_new_[i].maxCoeff() < 1e8)
    {
      u_ = u_new_[i];
      x_ = x_new_[i];
      c_ = c_new_[i];
      diverge_ = true;
      break;
    }
  }

  if (diverge_)
  {

  }

  for (size_t iter=1; iter < params_.max_iter; iter++)
  {
    // ====== STEP 1: differentiate dynamics and cost along new trajectory
    if(is_dynamics_changed_after_tick)
    {
      //#pragma omp parallel for
      for (size_t i = 0; i < N_; i++)
      {
        dynamicsCost(x_.col(i), u_.col(i), i, fx_[i], fu_[i],
                     cx_[i], cu_[i], cxx_[i], cxu_[i], cuu_[i]);

      }
    }

    // ====== STEP 2: backward pass, compute optimal control law and cost-to-go
    // Eq. (4) ~ (6)
    is_backpass_done = false;
    int backpass_diverge;
    while (!is_backpass_done)
    {
      backwardPass(lambda,backpass_diverge,Vx,Vxx,dV);
      if (backpass_diverge) {
        dlambda = max(dlambda * params_.lambda_factor, params_.lambda_factor);
        lambda = max(lambda * dlambda, params_.lambda_min);
        if (lambda > params_.lambda_max) {
          if(verbose_ >= vbl_warning)
          {
            cout << "Warning: Lambda caused divergence" << endl;
          }
          break;
        }
        continue;
      }
      is_backpass_done = true;
    }

    gnorm = 0.0;
    for (int i = 0; i < N_ - 1; i++)
      gnorm = gnorm + absMaxVector(l_[i], u_.col(i));

    gnorm = gnorm / (N_ - 1);
    if (gnorm < params_.tol_grad && lambda < 1e-5)
    {
      dlambda = min(dlambda / params_.lambda_factor, 1.0 / params_.lambda_factor);
      if (lambda > params_.lambda_min)
        lambda = lambda * dlambda;

      if(verbose_ >= vbl_warning)
      {
        cout << "Warning:: Break due to small gradient" << endl;
      }
      break;
    }

    // ====== STEP 3: line-search to find new control sequence, trajectory, cost
    is_forwardpass_done = false;
    if (is_backpass_done)
    {
      //#pragma omp parallel for
      for (size_t al_i = 0; al_i < k_; al_i++) // alpha
      {
        forwardPass(params_.alpha(al_i),x_new_[al_i], u_new_[al_i], c_new_[al_i]);
      }

      for (int i = 0; i < k_; i++)
        D_cost_(i) = c_.sum() - c_new_[i].sum();

      d_cost = D_cost_.maxCoeff(&w);
      alpha = params_.alpha(w);
      expected = -1.0*alpha*(dV(0) + alpha*dV(1));


      if (expected > 0)
        z = d_cost / expected;
      else {
        z = d_cost / abs(d_cost);
        if(verbose_ >= vbl_warning)
        {
          cout << "Warning:: non-positive expected reduction" << endl;
        }
      }
      if (z > params_.z_min)
        is_forwardpass_done = true;
      else
        alpha = 0.0;

    }

    if (is_forwardpass_done)
    {

      if(verbose_ >= vbl_verbose)
      {
        cout << "iteration:" << iter << " Cost " << c_.sum() << " dCost " << d_cost  <<
                " expected " << expected << " gradient " << gnorm <<endl;
      }
      c_ = c_new_[w];
      x_ = x_new_[w];
      u_ = u_new_[w];
      u0_ = u_;

      dlambda = min(dlambda / params_.lambda_factor, 1.0 / params_.lambda_factor);
      if (lambda > params_.lambda_min)
        lambda = lambda * dlambda;

      if (d_cost < params_.tol_fun) {
        if(verbose_ >= vbl_info)
        {
          cout << "Successful convergence" << endl;
        }
        break;
      }

      is_dynamics_changed_after_tick = true;
    }
    else
    {
      if(verbose_ >= vbl_verbose)
      {
        cout << "iteration:" << iter << " " << "NO STEP" << endl;
      }
      dlambda = max(dlambda * params_.lambda_factor, params_.lambda_factor);
      lambda = max(lambda * dlambda, params_.lambda_min);
      if (lambda > params_.lambda_max) {
        if(verbose_ >= vbl_warning)
        {
          cout << "Warning:: non-positive expected reduction" << endl;
        }
        break;
      }
    }

  }
}

#endif
