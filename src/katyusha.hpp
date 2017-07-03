#ifndef STOPT_KATYUSHA_HPP
#define STOPT_KATYUSHA_HPP

#include "svrg.hpp"

namespace stopt {
template <typename _Scalar>
class katyusha : public svrg<_Scalar> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;
  using v = svrg<_Scalar>;

 public:
  katyusha(const std::string &data_libsvm_format,
           const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~katyusha() {}

  void train();

 private:
  _Scalar tau1_;
  _Scalar tau2_;
  _Scalar alpha_;
  _Scalar dia_gamma_;

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum1_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum2_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum3_;
};

template <typename _Scalar>
katyusha<_Scalar>::katyusha(const std::string &name, const bool &rzv,
                            const bool &flag_info)
    : svrg<_Scalar>(name, rzv, flag_info) {
  s::prob_.setConstant(r::num_ins_, 1.0 / r::num_ins_);
  momemtum1_.setZero(r::num_fea_);
  momemtum2_.setZero(r::num_fea_);
  momemtum3_.setZero(r::num_fea_);
}

template <typename _Scalar>
void katyusha<_Scalar>::train() {
  info<> info_obj{};
  v::set_prob_data_driven();

  const uint64_t n = r::num_ins_;
  const uint64_t a = s::num_mbs_;
  const int n_a = static_cast<int>(n / a);

  tau2_ = 0.5 / a;
  tau1_ = tau2_ * std::min(1.0, std::sqrt(8.0 * n_a * a * r::lambda2_ /
                                          (3.0 * v::mean_gamma_)));
  dia_gamma_ = 0.5 * v::mean_gamma_ / (a * tau2_);
  alpha_ = 1.0 / (3.0 * tau1_ * dia_gamma_);

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> tmp_momemtum(r::num_fea_);
  const _Scalar theta =
      r::lambda2_ > 0.0 ? 1.0 + std::min(alpha_ * r::lambda2_, 0.25 / a) : 1.0;

  for (uint64_t itr = 0; itr < r::max_itr_; ++itr) {
    bool stop = r::check_stopping_criteria();
    if (r::flag_info_)
      info_obj.out_time(itr, r::duality_gap_, r::primal_obj_value_,
                        r::dual_obj_value_);
    if (stop) break;

    if (r::lambda2_ == 0.0) {
      tau1_ = static_cast<_Scalar>(itr + 1) / (itr + 5);
      alpha_ = 1.0 / (3.0 * tau1_ * v::mean_gamma_);
    }
    _Scalar sum_theta = 0.0, thetaj = 1.0 / theta;
    tmp_momemtum.setZero();
    v::calc_full_loss_grad(momemtum1_);
    for (int iitr = 0; iitr < n_a; ++iitr) {
      r::primal_var_ = tau1_ * momemtum3_ + tau2_ * momemtum1_ +
                       (1.0 - tau2_ - tau1_) * momemtum2_;
      s::sample();
      v::calc_variance_reduced_loss_grad(r::primal_var_);
      r::prox_regularized(alpha_, momemtum3_, v::vr_loss_grad_, momemtum3_);
      r::prox_regularized(1.0 / (3.0 * dia_gamma_), r::primal_var_,
                          v::vr_loss_grad_, momemtum2_);
      thetaj *= theta;
      sum_theta += thetaj;
      tmp_momemtum += thetaj * momemtum2_;
    }
    momemtum1_ = (1.0 / sum_theta) * tmp_momemtum;
  }
}
}
#endif
