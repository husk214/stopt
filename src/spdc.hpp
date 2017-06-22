#ifndef STOPT_SPDC_HPP
#define STOPT_SPDC_HPP

#include "stochastic.hpp"

namespace stopt {
template <typename _Scalar>
class spdc : public stochastic<_Scalar, Eigen::RowMajor> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;

 public:
  spdc(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~spdc() {}

  void train(const sampling sampling_type = sampling::data_driven);

 private:
  _Scalar theta_;
  _Scalar tau_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> sigmas_;

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum_w_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> pre_dual_var_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum_a_;

  void truncate_prob();
  void set_prob_data_driven();
  void set_prob_optimality_violation();

  void set_sigmas();
  void set_spdc_parameters();
  void set_adaspdc_parameters();
};

template <typename _Scalar>
spdc<_Scalar>::spdc(const std::string &name, const bool &rzv,
                    const bool &flag_info)
    : stochastic<_Scalar, Eigen::RowMajor>(name, rzv, flag_info) {
  sigmas_.setZero(r::num_ins_);
  momemtum_w_ = r::primal_var_;
  pre_dual_var_ = r::dual_var_;
  momemtum_a_.setZero(r::num_fea_);
}

template <typename _Scalar>
void spdc<_Scalar>::train(const sampling sam) {
  info<> info_obj{};
  if (sam == sampling::data_driven) set_prob_data_driven();
  set_spdc_parameters();

  const uint64_t n = r::num_ins_;
  const uint64_t a = s::num_mbs_;
  const _Scalar one_n = 1.0 / r::num_ins_;
  const int n_a = static_cast<int>(n / a);

  _Scalar tmp = 0.0;
  for (uint64_t itr = 1; itr < r::max_itr_; ++itr) {
    bool stop = r::check_stopping_criteria();
    if (r::flag_info_)
      info_obj.out_time(itr, r::duality_gap_, r::primal_obj_value_,
                        r::dual_obj_value_);
    if (stop) break;
    if (sam == sampling::optimality_violation) set_prob_optimality_violation();
    for (int iitr = 0; iitr < n_a; ++iitr) {
      s::sample();
      if (sam == sampling::optimality_violation) set_adaspdc_parameters();
      for (auto i : s::selected_idx_)
        s::prox_loss_conj(sigmas_[i] / (s::prob_[i] * n), momemtum_w_, i);
      momemtum_a_.noalias() = -r::yxa_n_;
      for (auto i : s::selected_idx_) {
        switch (r::problem_type_) {
          case problem_type::classification:
            tmp = one_n * r::y_[i] * (r::dual_var_[i] - pre_dual_var_[i]) *
                  (1.0 / (a * s::prob_[i] * n) - 1.0);
            break;
          case problem_type::regression:
            tmp = one_n * (r::dual_var_[i] - pre_dual_var_[i]) *
                  (1.0 / (a * s::prob_[i] * n) - 1.0);
            break;
        }
        for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
          momemtum_a_[it.index()] -= tmp * it.value();
        pre_dual_var_[i] = r::dual_var_[i];
      }
      momemtum_w_.noalias() = -theta_ * r::primal_var_;
      r::prox_regularized(tau_, r::primal_var_, momemtum_a_, r::primal_var_);
      momemtum_w_.noalias() += (1.0 + theta_) * r::primal_var_;
    }
  }
}

template <typename _Scalar>
void spdc<_Scalar>::truncate_prob() {
  _Scalar sum = 0.0;
  for (auto i : r::active_instance_) sum += s::prob_[i];
  s::prob_ /= sum;

  const _Scalar p_max = s::prob_.maxCoeff();
  const _Scalar one_asn = 1.0 / (s::num_mbs_ * std::sqrt(r::num_act_ins_));
  if (p_max > one_asn) {
    const _Scalar one_n = 1.0 / r::num_act_ins_;
    const _Scalar zeta = (one_asn - one_n) / (p_max - one_n);
    s::prob_ *= zeta;
    s::prob_ += (1.0 - zeta) * one_n;
  }
}

template <typename _Scalar>
void spdc<_Scalar>::set_prob_data_driven() {
  const _Scalar tmp =
      std::sqrt(r::num_ins_ * r::lambda_ * r::gamma_) * s::num_mbs_;
  s::prob_ = tmp + r::xi_l2norm_;
  truncate_prob();
}

// return min_i {p_i}
template <typename _Scalar>
void spdc<_Scalar>::set_prob_optimality_violation() {
  _Scalar tmp = 0.0, vio_min = std::numeric_limits<_Scalar>::max();
  switch (r::loss_term_) {
    case loss_term::smoothed_hinge:
      for (auto i : r::active_instance_) {
        tmp = -r::dual_var_[i] * r::y_[i];
        if (r::margin_[i] < 0.0) {
          s::prob_[i] = std::abs(tmp);
        } else if (r::margin_[i] > r::gamma_) {
          s::prob_[i] = std::abs(tmp + r::y_[i]);
        } else {
          s::prob_[i] = std::abs(tmp + r::y_[i] * r::margin_[i] / r::gamma_);
        }
        if (s::prob_[i] > 0.0) vio_min = std::min(vio_min, s::prob_[i]);
      }
      break;
    case loss_term::squared_hinge:
      for (auto i : r::active_instance_) {
        tmp = -r::dual_var_[i] * r::y_[i];
        if (r::margin_[i] < 0.0) {
          s::prob_[i] = std::abs(tmp);
        } else {
          s::prob_[i] = std::abs(tmp + r::y_[i] * r::margin_[i]);
        }
        if (s::prob_[i] > 0.0) vio_min = std::min(vio_min, s::prob_[i]);
      }
      break;
    case loss_term::logistic:
      break;
    case loss_term::smoothed_insensitve:
      break;
    case loss_term::squared:
      break;
  }

  for (auto i : r::active_instance_) {
    s::prob_[i] += vio_min;
    s::prob_[i] *= r::xi_l2norm_[i];
  }
  truncate_prob();
  set_sigmas();
  std::discrete_distribution<> dst(s::prob_.data(),
                                   s::prob_.data() + s::prob_.size());
  s::sampling_dst_ = std::move(dst);
}

template <typename _Scalar>
void spdc<_Scalar>::set_sigmas() {
  if (s::num_mbs_ < std::sqrt(r::num_ins_)) {
    sigmas_ =
        (0.5 * r::num_ins_ * std::sqrt(r::num_ins_ * r::gamma_ / r::lambda2_)) *
        (s::prob_ / r::xi_l2norm_);
  } else {
    sigmas_ = 0.5 * r::num_ins_ * std::sqrt(r::lambda2_ / r::gamma_) /
              (r::max_xi_norm_ * s::num_mbs_);
  }
}

template <typename _Scalar>
void spdc<_Scalar>::set_spdc_parameters() {
  const double n = r::num_ins_;
  const double a = static_cast<double>(s::num_mbs_);
  const _Scalar tmp = 0.5 * n * std::sqrt(n * r::gamma_ / r::lambda2_);
  set_sigmas();
  if (a < std::sqrt(r::num_ins_)) {
    tau_ = sigmas_.minCoeff() * (a / n);
    theta_ = 1.0 - 1.0 / (1.0 / (a * s::prob_) + tmp / sigmas_).maxCoeff();
  } else {
    tau_ = 0.5 * std::sqrt(r::gamma_ / r::lambda2_) / r::max_xi_norm_;
    theta_ =
        1.0 -
        1.0 / (n / a + r::max_xi_norm_ / std::sqrt(r::lambda2_ * r::gamma_));
  }
}

template <typename _Scalar>
void spdc<_Scalar>::set_adaspdc_parameters() {
  const _Scalar n = static_cast<_Scalar>(r::num_ins_);
  const _Scalar tmp = 0.5 * n * std::sqrt(n * r::gamma_ / r::lambda2_);
  _Scalar min_sig = std::numeric_limits<_Scalar>::max();
  _Scalar tmp1 = 0.0;
  for (auto i : s::selected_idx_) {
    min_sig = std::min(min_sig, sigmas_[i]);
    tmp1 = std::max(tmp1, 1.0 / (s::num_mbs_ * s::prob_[i]) + tmp / sigmas_[i]);
  }
  tau_ = min_sig * (s::num_mbs_ / n);
  theta_ = 1.0 - 1.0 / tmp1;
}
}
#endif
