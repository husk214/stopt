#ifndef STOPT_SDCA_HPP
#define STOPT_SDCA_HPP

#include "stochastic.hpp"

namespace stopt {
template <typename _Scalar>
class sdca : public stochastic<_Scalar, Eigen::RowMajor> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;

 public:
  sdca(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~sdca() {}

  void train();

 protected:
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> xi_sqnorm_;
};

template <typename _Scalar>
sdca<_Scalar>::sdca(const std::string &name, const bool &rzv,
                    const bool &flag_info)
    : stochastic<_Scalar, Eigen::RowMajor>(name, rzv, flag_info) {
  xi_sqnorm_ = r::xi_l2norm_.square();
}

template <typename _Scalar>
void sdca<_Scalar>::train() {
  info<> info_obj{};

  const uint64_t n = r::num_ins_;
  const int n_a = static_cast<int>(n / s::num_mbs_);
  const _Scalar lambda2_n = r::lambda2_ * n;
  const _Scalar tau =
      0.5 * std::sqrt(r::gamma_ / r::lambda2_) / r::max_xi_norm_;

  for (uint64_t itr = 1; itr < r::max_itr_; ++itr) {
    bool stop = r::check_stopping_criteria();
    if (r::flag_info_)
      info_obj.out_time(itr, r::duality_gap_, r::primal_obj_value_,
                        r::dual_obj_value_);
    if (stop) break;
    for (int iitr = 0; iitr < n_a; ++iitr) {
      s::sample();
      for (auto i : s::selected_idx_)
        s::prox_loss_conj(lambda2_n / xi_sqnorm_[i], r::primal_var_, i);
      r::prox_regularized(tau, r::primal_var_, -r::yxa_n_, r::primal_var_);
      // r::set_primal_var_kkt();
    }
  }
}
}
#endif
