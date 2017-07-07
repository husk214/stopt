#ifndef STOPT_CATALYST_HPP
#define STOPT_CATALYST_HPP

#include "rerm.hpp"
#include "stochastic.hpp"

namespace stopt {

template <class _Alg, class _Scalar>
void catalyst(_Alg &alg, const int max_itr = 1000) {
  info<> info_obj{};
  alg.set_perturbed_algorithm(perturbed_algorithm::catalyst);
  regularization_term original_reg_type = alg.get_regularization_type();
  alg.set_flag_info(false);
  _Scalar kappa = 0.0, q = 0.0, eps = 0.0, alpha = 0.0, rho = 0.0;
  if (alg.get_regularization_type() == regularization_term::l1) {
    alg.set_regularization_term(regularization_term::elastic_net);
    kappa = alg.get_lambda1();
    alpha = 0.5 * (std::sqrt(5) - 1.0);
  } else {
    kappa = alg.get_lambda2();
    q = alg.get_lambda2() / (alg.get_lambda2() + kappa);
    rho = 0.9 * std::sqrt(q);
    alpha = (1.0 - q) * (1.0 - q) ;
  }


  for (int itr = 0; itr < max_itr; ++itr) {
    alg.set_catalyst_kappa(kappa);
    alg.set_stopping_criteria(alg.calc_primal_obj_value());
    alg.train();
    alg.set_regularization_term(regularization_term::l1);
    alg.calc_duality_gap();
    info_obj.out_time(itr, alg.get_total_epoch(), alg.get_duality_gap(),
                      alg.get_primal_obj_value(), alg.get_dual_obj_value(), mu);
    mu *= 0.5;
  }
}
}  // namespace stopt
#endif
