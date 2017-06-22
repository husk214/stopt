#ifndef STOPT_ADAPTREG_HPP
#define STOPT_ADAPTREG_HPP

#include "rerm.hpp"
#include "stochastic.hpp"

namespace stopt {

template <typename _Alg>
class adaptreg {
  using r = rerm<double, Eigen::RowMajor>;
  using s = stochastic<double, Eigen::RowMajor>;

 public:
  adaptreg() {}
  ~adaptreg() {}

  void train(_Alg &a);
};

template <typename _Alg>
void adaptreg<_Alg>::train(_Alg &alg) {
  info<> info_obj{};
  alg.set_regularization_term(regularization_term::elastic_net);
  double mu = 0.01;
  alg.set_lambda2(mu);
  double ppv = alg.calc_primal_obj_value();
  alg.set_dual_var_kkt();
  double mpv = alg.calc_dual_obj_value();
  int iitr = 0;
  for (int itr = 0; itr < 1000; ++itr) {
    double pre_ppv = alg.calc_primal_obj_value();
    info_obj.out("outer", ppv, mpv, ppv-mpv);
    while (ppv - mpv > 0.25 * (pre_ppv - mpv)) {
      mpv = alg.calc_dual_obj_value();
      alg.set_dual_var_kkt();
      ppv = alg.calc_primal_obj_value();
      info_obj.out(iitr++, ppv - 0.5 * mu * alg.calc_primal_var_sqnorm(), mpv, ppv-mpv);
    }
    mu *= 0.5;
    alg.set_lambda2(mu);
  }
}
}
#endif
