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
  regularization_term perturbed_reg_type = regularization_term::elastic_net;
  alg.set_flag_info(false);
  _Scalar catalyst_a = 0.5, catalyst_b = 0.5;
  _Scalar q = 0.0, alpha = 0.0, rho = 0.0, beta = 0.0;
  _Scalar kappa = catalyst_a * (1.0 / alg.get_gamma() - alg.get_lambda2()) /
                      (alg.get_num_ins() + catalyst_b) -
                  alg.get_lambda2();
  if (alg.get_regularization_type() == regularization_term::l1) {
    alg.set_regularization_term(regularization_term::elastic_net);
    alpha = 0.5 * (std::sqrt(5) - 1.0);
  } else {
    q = alg.get_lambda2() / (alg.get_lambda2() + kappa);
    rho = 0.9 * std::sqrt(q);
    alpha = 1.2 * std::sqrt(q);
    if (alg.get_regularization_type() == regularization_term::squared)
      perturbed_reg_type = regularization_term::squared;
  }
  info_obj.out("kappa:", kappa);
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> ref = alg.get_primal_var();
  _Scalar origin_stopping_criteria = alg.get_stopping_criteria();
  _Scalar origin_duality_gap = 0.5;
  _Scalar tmp = 0.0;
  _Scalar eta = 0.1;
  alg.set_stopping_criteria(1.0 / 9.0);
  alg.set_catalyst_kappa(kappa);
  for (int itr = 0;
       itr < max_itr && origin_duality_gap > origin_stopping_criteria;
       ++itr) {
    ref = alg.get_primal_var();
    alg.train();
    alg.set_regularization_term(original_reg_type);
    alg.set_perturbed_algorithm(perturbed_algorithm::none);
    alg.calc_duality_gap();
    if (alg.get_lambda2() > 0.0) {
      alg.set_stopping_criteria(std::pow(1.0 - rho, itr + 1) / 9.0);
    } else {
      alg.set_stopping_criteria(1.0 / (9.0 * std::pow(itr + 2, 4.0 + eta)));
    }
    beta = alpha;
    tmp = q - alpha * alpha;
    alpha = 0.5 * (tmp + std::sqrt(tmp * tmp + 4.0 * alpha * alpha));
    beta *= (1.0 - beta) / (beta * beta + alpha);
    ref = (1.0 + beta) * alg.get_primal_var() - beta * ref;
    alg.set_catalyst_ref(ref);
    alg.set_regularization_term(perturbed_reg_type);
    alg.set_perturbed_algorithm(perturbed_algorithm::catalyst);
    origin_duality_gap = alg.get_duality_gap();
    info_obj.out_time(itr, alg.get_total_epoch(), alg.get_duality_gap(),
                      alg.get_primal_obj_value(), alg.get_dual_obj_value(), alpha);
  }
}
}  // namespace stopt
#endif
