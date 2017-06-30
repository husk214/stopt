#ifndef STOPT_ADAPTREG_HPP
#define STOPT_ADAPTREG_HPP

#include "rerm.hpp"
#include "stochastic.hpp"

namespace stopt {

template <class _Alg, class _Scalar>
void adaptreg(_Alg &alg, const int max_itr = 1000) {
  info<> info_obj{};
  alg.set_perturbed_algorithm(perturbed_algorithm::adaptreg);
  alg.set_flag_info(false.);
  _Scalar mu = 1.0;

  for (int itr = 0; itr < max_itr; ++itr) {
    alg.set_regularization_term(regularization_term::elastic_net);
    alg.set_lambda2(mu);
    alg.set_stopping_criteria(alg.calc_primal_obj_value());
    alg.train();
    alg.set_regularization_term(regularization_term::l1);
    alg.calc_duality_gap();
    info_obj.out_time(itr, alg.get_total_epoch(), alg.get_duality_gap(),
                      alg.get_primal_obj_value(), alg.get_dual_obj_value(), mu);
    mu *= 0.5;
  }
}
}
#endif
