#ifndef STOPT_SAGA_HPP
#define STOPT_SAGA_HPP

#include "stochastic.hpp"

namespace stopt {
template <typename _Scalar>
class saga : public svrg<_Scalar> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;
  using v = svrg<_Scalar>;

 public:
  saga(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~saga() {}

  void train();

 protected:
  void calc_variance_reduced_loss_grad(
      const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &v);
  void update_variance_reduced_loss_grad(const _Scalar dif, const uint64_t i);
};

template <typename _Scalar>
saga<_Scalar>::saga(const std::string &name, const bool &rzv,
                    const bool &flag_info)
    : svrg<_Scalar>(name, rzv, flag_info) {}

template <typename _Scalar>
void saga<_Scalar>::train() {
  info<> info_obj{};

  const _Scalar eta = 0.5 / (r::max_xi_norm_ * r::gamma_);
  const int n_a = static_cast<int>(r::num_ins_ / s::num_mbs_);

  v::calc_full_loss_grad(r::primal_var_);
  for (uint64_t itr = 1; itr < r::max_itr_; ++itr) {
    bool stop = r::check_stopping_criteria();
    if (r::flag_info_)
      info_obj.out_time(itr, r::duality_gap_, r::primal_obj_value_,
                        r::dual_obj_value_);
    if (stop) break;
    for (int iitr = 0; iitr < n_a; ++iitr) {
      s::sample();
      v::calc_variance_reduced_loss_grad(r::primal_var_, true);
      r::prox_regularized(eta, r::primal_var_, v::vr_loss_grad_,
                          r::primal_var_);
    }
  }
}
}
#endif
