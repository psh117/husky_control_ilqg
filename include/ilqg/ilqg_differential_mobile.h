/**
  @author Suhan Park
  */

#include "ilqg.h"
#include "pseudo_huber.h"

constexpr unsigned int STATE_DIMENSION = 5;
constexpr unsigned int INPUT_DIMENSION = 2;

typedef double data_type;
// State
// x
// y
// theta
// w_left
// w_right

// Control Input
// 0    0
// 0    0
// 0    0
// 1    0
// 0    1

/**
 * @brief The iLQGDifferentialMobile class
 */
class iLQGDifferentialMobile : public iLQG<data_type, STATE_DIMENSION, INPUT_DIMENSION>
{

public:
  iLQGDifferentialMobile(double L = 2.0, double h = 0.03) :
    iLQG<data_type, STATE_DIMENSION, INPUT_DIMENSION>(), L_(L), h_(h) {}
private:
  double L_; ///< Distance between left and right wheel axis
  double h_; ///< timestep (seconds)

private:
  //VIRTUAL
  virtual const VectorX dynamicsCalc(const VectorX& x, const VectorU& u)
  {
    VectorX Y;

    double ar = u(0);
    double al = u(1);

    double o = x(2);
    double vl =x(3);
    double vr =x(4);

    double v = h_ * (vr+vl)/2.;
    double w = h_ * (vr-vl)/L_;

    Y = x;
    Y(0) += v * cos(o);
    Y(1) += v * sin(o);
    Y(2) += w;
    Y(3) += ar*h_;
    Y(4) += al*h_;

    return Y;
  }

  virtual double costFunction(const VectorX& x, const VectorU& u, size_t index)
  {
    double lf, lx, lu; // Final, running, control cost
    double total_cost;   // total cost

    VectorU cu;
    cu << 1e-2 * .01, 1e-2 * .01;

    VectorX cf;
    cf << 0.5, 0.5, 1., .1, .1;

    VectorX pf;
    pf << .01, .01, .01, .1, .1;

    Vector3d cx;
    cx << 1e-1, 1e-1, 5e-2;

    Vector3d px = 0.1*Vector3d::Ones();

    // Control cost
    lu = cu.transpose() * VectorU(u.array().square());

    // Final cost
    if(index == getHorizon() - 1)
    {
      lf = cf.transpose() * sabs<VectorX>(x-xd_,pf);
    }
    else
    {
      lf = 0;
    }

    // Running cost
    lx = cx.transpose() * sabs<Vector3d>((x-xd_).head<3>(), px);

    total_cost = lu + lx + lf;
    return total_cost;
  }
};
