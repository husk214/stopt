#ifndef STOPT_DASVRDA_HPP
#define STOPT_DASVRDA_HPP

#include "svrg.hpp"

namespace stopt {
template <typename _Scalar>
class dasvrda : public svrg<_Scalar> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;
  using v = svrg<_Scalar>;

 public:
  dasvrda(const std::string &data_libsvm_format,
          const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~dasvrda() {}

  void train();

 private:
  _Scalar eta_;
  _Scalar theta1_;
  _Scalar theta2_;

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> avg_loss_grad_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum1_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum2_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum3_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum03_;
};

template <typename _Scalar>
dasvrda<_Scalar>::dasvrda(const std::string &name, const bool &rzv,
                          const bool &flag_info)
    : svrg<_Scalar>(name, rzv, flag_info) {
  avg_loss_grad_.setZero(r::num_fea_);
  momemtum1_.setZero(r::num_fea_);
  momemtum2_.setZero(r::num_fea_);
  momemtum3_.setZero(r::num_fea_);
  momemtum03_.setZero(r::num_fea_);
}

template <typename _Scalar>
void dasvrda<_Scalar>::train() {
  info<> info_obj{};
  v::set_prob_data_driven();

  const uint64_t n = r::num_ins_;
  const uint64_t a = s::num_mbs_;
  const int n_a = static_cast<int>(n / a);

  uint64_t U = r::max_itr_;
  if (r::lambda2_ > 0.0) {
    U = 1.0 + std::ceil(std::sqrt(1.0 / (v::mean_gamma_ * r::lambda2_)) *
                        (static_cast<_Scalar>(a) / n + 1.0 / std::sqrt(n)));
  }

  const _Scalar gamma = 0.5 * (3.0 + std::sqrt(9.0 + 8.0 * a / (n_a)));
  eta_ = 1.0 / ((1.0 + gamma * (n_a + 1) / a) * v::mean_gamma_);
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> pre_momemtum1 = momemtum1_;
  bool stop = false;
  for (uint64_t t = 1; t <= r::max_itr_ && !stop; ++t) {
    momemtum3_ = momemtum1_;
    _Scalar pre_theta1 = 0.0;
    for (uint64_t u = 1; u <= U; ++u) {
      stop = r::check_stopping_criteria();
      if (r::flag_info_)
        info_obj.out_time(U * (t - 1) + u, r::duality_gap_,
                          r::primal_obj_value_, r::dual_obj_value_);
      if (stop) break;
      pre_momemtum1 = momemtum1_;
      theta1_ = 0.5 * (1.0 - 1.0 / gamma) * (u + 1);
      momemtum2_ =
          momemtum1_ +
          ((pre_theta1 - 1.0) / theta1_) * (momemtum1_ - pre_momemtum1) +
          (pre_theta1 / theta1_) * (momemtum3_ - momemtum1_);
      _Scalar pre_theta2 = 0.5;
      v::calc_full_loss_grad(momemtum1_);
      avg_loss_grad_.setZero();
      r::primal_var_ = momemtum2_;
      momemtum03_ = momemtum2_;
      momemtum3_ = momemtum2_;
      for (uint64_t k = 1; k <= n_a; ++k) {
        theta2_ = (0.5 * (k + 1));
        momemtum2_ = (1.0 - 1.0 / theta2_) * r::primal_var_ +
                     (1.0 / theta2_) * momemtum3_;
        s::sample();
        v::calc_variance_reduced_loss_grad(momemtum2_);
        avg_loss_grad_ *= (1.0 - 1.0 / theta2_);
        avg_loss_grad_ += (1.0 / theta2_) * v::vr_loss_grad_;
        r::prox_regularized(eta_ * theta2_ * pre_theta2, momemtum03_,
                            avg_loss_grad_, momemtum3_);
        r::primal_var_ = (1.0 - 1.0 / theta2_) * r::primal_var_ +
                         (1.0 / theta2_) * momemtum3_;
        pre_theta2 = theta2_;
      }
      momemtum1_ = r::primal_var_;
      pre_theta1 = theta1_;
    }
  }
}
}
#endif
