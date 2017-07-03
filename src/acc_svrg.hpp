#ifndef STOPT_ACC_SVRG_HPP
#define STOPT_ACC_SVRG_HPP

#include "stochastic.hpp"
#include "svrg.hpp"

namespace stopt {
template <typename _Scalar>
class acc_svrg : public svrg<_Scalar> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;
  using v = svrg<_Scalar>;

 public:
  acc_svrg(const std::string &data_libsvm_format,
           const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~acc_svrg() {}

  void train();

 protected:
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum1_;
  _Scalar beta_;
};

template <typename _Scalar>
acc_svrg<_Scalar>::acc_svrg(const std::string &name, const bool &rzv,
                            const bool &flag_info)
    : svrg<_Scalar>(name, rzv, flag_info), momemtum1_(r::num_fea_) {}

template <typename _Scalar>
void acc_svrg<_Scalar>::train() {
  info<> info_obj{};
  const _Scalar eta = 0.5 / (r::max_xi_norm_ * r::gamma_);
  const int n_a = static_cast<int>(r::num_ins_ / s::num_mbs_);
  beta_ = (1.0 - std::sqrt(eta * r::lambda2_)) /
          (1.0 / std::sqrt(eta * r::lambda2_));
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> tmpw(r::num_fea_);
  for (uint64_t itr = 0; itr < r::max_itr_; ++itr) {
    bool stop = r::check_stopping_criteria();
    if (r::flag_info_)
      info_obj.out_time(itr, r::duality_gap_, r::primal_obj_value_,
                        r::dual_obj_value_);
    if (stop) break;
    v::calc_full_loss_grad(r::primal_var_);
    momemtum1_.noalias() = r::primal_var_;
    for (uint64_t iitr = 0; iitr < n_a; ++iitr) {
      s::sample();
      v::calc_variance_reduced_loss_grad(momemtum1_);
      tmpw.noalias() = r::primal_var_;
      r::prox_regularized(eta, momemtum1_, v::vr_loss_grad_, r::primal_var_);
      momemtum1_.noalias() = r::primal_var_ + beta_ * (r::primal_var_ - tmpw);
    }
  }
}
}
#endif
