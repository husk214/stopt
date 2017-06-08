#ifndef STOPT_SPDC_HPP
#define STOPT_SPDC_HPP

#include "rerm.hpp"
#include <unordered_set>

namespace stopt {
template <typename _Scalar> class spdc : public rerm<_Scalar, Eigen::RowMajor> {
  using r = rerm<_Scalar, Eigen::RowMajor>;

public:
  spdc(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~spdc() {}

  void train(const sampling sampling_type = sampling::data_driven);

private:
  _Scalar theta_;
  _Scalar tau_;

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum_w_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum_a_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> momemtum_aa_;

  Eigen::Array<_Scalar, Eigen::Dynamic, 1> sigmas_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> prob_;
  std::unordered_set<uint64_t> selected_dual_coordinates_;

  std::mt19937 generator_;
  std::discrete_distribution<> sampling_dst_;

  void update_dual();
  void update_dual_momemtum(_Scalar delta_alpha_i, uint64_t i);
  void update_dual_smoothed_hinge();
  void update_dual_squared_hinge();

  void update_primal();
  void update_primal_momemtum();
  void update_primal_elastic_net();
  void update_primal_squared();

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
    : rerm<_Scalar, Eigen::RowMajor>(name, rzv, flag_info) {
  sigmas_.setZero(r::num_ins_);
  prob_.setConstant(r::num_ins_, 1.0 / r::num_ins_);
  momemtum_w_ = r::primal_var_;
  momemtum_a_.setZero(r::num_fea_);
  momemtum_aa_.setZero(r::num_fea_);
  // ToDO: imporve
  std::random_device rd;
  std::mt19937 g(rd());
  generator_ = std::move(g);
  std::discrete_distribution<> dst(prob_.data(), prob_.data() + prob_.size());
  sampling_dst_ = std::move(dst);
}

template <typename _Scalar> void spdc<_Scalar>::train(const sampling sam) {
  info<> info_obj{};
  if (sam == sampling::data_driven)
    set_prob_data_driven();
  set_spdc_parameters();

  selected_dual_coordinates_.reserve(r::num_mbs_);

  const uint64_t n = r::num_ins_;
  const int n_a = static_cast<int>(n / r::num_mbs_);

  _Scalar pv = r::calc_primal_obj_value();
  _Scalar dv = r::calc_dual_obj_value();
  _Scalar duality_gap = pv - dv;
  if (r::flag_info_)
    info_obj.out_time(0, duality_gap, pv, dv, prob_.sum());
  for (uint64_t itr = 1; itr < r::max_itr_ && duality_gap > r::stop_criterion_;
       ++itr) {

    for (int iitr = 0; iitr < n_a; ++iitr) {
      for (int count = 0; count < r::num_mbs_; ++count)
        selected_dual_coordinates_.insert(sampling_dst_(generator_));
      if (sam != sampling::uniform)
        set_adaspdc_parameters();
      update_dual();
      update_primal();
    }

    if (itr % 1 == 0) {
      pv = r::calc_primal_obj_value();
      dv = r::calc_dual_obj_value();
      if (r::flag_info_)
        info_obj.out_time(itr, pv - dv, pv, dv);
      if (sam == sampling::optimality_violation) {
        set_prob_optimality_violation();
        std::discrete_distribution<> dst(prob_.data(),
                                         prob_.data() + prob_.size());
        sampling_dst_ = std::move(dst);
      }
    }
  }
}

template <typename _Scalar> void spdc<_Scalar>::update_dual() {
  switch (r::loss_term_) {
  case loss_term::smoothed_hinge:
    update_dual_smoothed_hinge();
    break;
  case loss_term::squared_hinge:
    update_dual_squared_hinge();
    break;
  }
  selected_dual_coordinates_.clear();
}

template <typename _Scalar> void spdc<_Scalar>::update_dual_smoothed_hinge() {
  const uint64_t n = r::num_ins_;
  _Scalar yi, tmp, beta_i, alpha_i_new, delta_alpha_i;
  for (const auto &i : selected_dual_coordinates_) {
    yi = r::y_[i];
    tmp = 0.0;
    for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
      tmp += momemtum_w_.coeff(it.index()) * it.value();
    beta_i = (sigmas_[i] * (yi * tmp - 1.0) - n * prob_[i] * r::dual_var_[i]) /
             (-sigmas_[i] * r::gamma_ - n * prob_[i]);
    alpha_i_new = std::min(1.0, std::max(0.0, beta_i));
    update_dual_momemtum(yi * (alpha_i_new - r::dual_var_[i]), i);
    r::dual_var_[i] = alpha_i_new;
  }
}

template <typename _Scalar> void spdc<_Scalar>::update_dual_squared_hinge() {
  const uint64_t n = r::num_ins_;
  _Scalar yi, tmp, beta_i, alpha_i_new, delta_alpha_i;
  for (const auto &i : selected_dual_coordinates_) {
    yi = r::y_[i];
    tmp = 0.0;
    for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
      tmp += momemtum_w_.coeff(it.index()) * it.value();
    beta_i = (sigmas_[i] * (yi * tmp - 1.0) - n * prob_[i] * r::dual_var_[i]) /
             (-sigmas_[i] - n * prob_[i]);
    alpha_i_new = std::max(0.0, beta_i);
    update_dual_momemtum(yi * (alpha_i_new - r::dual_var_[i]), i);
    r::dual_var_[i] = alpha_i_new;
  }
}

// ToDo improve
template <typename _Scalar>
void spdc<_Scalar>::update_dual_momemtum(_Scalar delta_alpha_i, uint64_t i) {
  if (delta_alpha_i != 0.0)
    for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it) {
      momemtum_a_[it.index()] += delta_alpha_i * it.value();
      momemtum_aa_[it.index()] +=
          delta_alpha_i * it.value() / (r::num_mbs_ * (prob_[i] * r::num_ins_));
    }
}

// ToDO
template <typename _Scalar> void spdc<_Scalar>::update_primal() {
  switch (r::regularization_term_) {
  case regularization_term::elastic_net:
    update_primal_elastic_net();
    break;
  case regularization_term::squared:
    update_primal_squared();
    break;
  }
}

// ToDO
template <typename _Scalar> void spdc<_Scalar>::update_primal_elastic_net() {
  const _Scalar one_n = 1.0 / static_cast<_Scalar>(r::num_ins_);
  _Scalar tmp_w, eta;
  const _Scalar tmp = tau_ * r::lambda1_;
  const _Scalar tmp1 = tau_ * r::lambda2_ + 1.0;
  for (auto j : r::active_feature_) {
    tmp_w = r::primal_var_[j] + tau_ * (r::yxa_n_[j] + momemtum_aa_[j]);
    if (tmp_w > tmp) {
      eta = (tmp_w - tmp) / tmp1;
    } else if (tmp_w < -tmp) {
      eta = (tmp_w + tmp) / tmp1;
    } else {
      eta = 0.0;
    }
    momemtum_w_[j] = eta + theta_ * (eta - r::primal_var_[j]);
    r::primal_var_[j] = eta;
    r::yxa_n_[j] += one_n * momemtum_a_[j];
    momemtum_a_[j] = 0.0;
    momemtum_aa_[j] = 0.0;
  }
}

// ToDO
template <typename _Scalar> void spdc<_Scalar>::update_primal_squared() {
  const _Scalar one_n = 1.0 / static_cast<_Scalar>(r::num_ins_);
  const _Scalar tmp1 = tau_ * r::lambda2_ + 1.0;
  _Scalar eta = 0.0;
  for (auto j : r::active_feature_) {
    eta =
        (r::primal_var_[j] + tau_ * (r::yxa_n_[j] + momemtum_aa_[j])) / tmp1;
    momemtum_w_[j] = eta + theta_ * (eta - r::primal_var_[j]);
    r::primal_var_[j] = eta;
    r::yxa_n_[j] += one_n * momemtum_a_[j];
    momemtum_a_[j] = 0.0;
    momemtum_aa_[j] = 0.0;
  }
}

template <typename _Scalar> void spdc<_Scalar>::truncate_prob() {
  _Scalar sum = 0.0;
  for (auto i : r::active_instance_)
    sum += prob_[i];
  prob_ /= sum;

  const _Scalar p_max = prob_.maxCoeff();
  const _Scalar one_asn = 1.0 / (r::num_mbs_ * std::sqrt(r::num_act_ins_));
  if (p_max > one_asn) {
    const _Scalar one_n = 1.0 / r::num_act_ins_;
    const _Scalar zeta = (one_asn - one_n) / (p_max - one_n);
    prob_ *= zeta;
    prob_ += (1.0 - zeta) * one_n;
  }
}

template <typename _Scalar> void spdc<_Scalar>::set_prob_data_driven() {
  const _Scalar tmp =
      std::sqrt(r::num_ins_ * r::lambda_) * std::sqrt(r::gamma_ * r::num_mbs_);
  prob_ = tmp + r::xi_l2norm_;
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
        prob_[i] = std::abs(tmp);
      } else if (r::margin_[i] > r::gamma_) {
        prob_[i] = std::abs(tmp + r::y_[i]);
      } else {
        prob_[i] = std::abs(tmp + r::y_[i] * r::margin_[i] / r::gamma_);
      }
      if (prob_[i] > 0.0)
        vio_min = std::min(vio_min, prob_[i]);
    }
    break;
  case loss_term::squared_hinge:
    for (auto i : r::active_instance_) {
      tmp = -r::dual_var_[i] * r::y_[i];
      if (r::margin_[i] < 0.0) {
        prob_[i] = std::abs(tmp);
      } else {
        prob_[i] = std::abs(tmp + r::y_[i] * r::margin_[i]);
      }
      if (prob_[i] > 0.0)
        vio_min = std::min(vio_min, prob_[i]);
    }
    break;
  }

  for (auto i : r::active_instance_) {
    prob_[i] += vio_min;
    prob_[i] *= r::xi_l2norm_[i];
  }
  truncate_prob();
  set_sigmas();
}

template <typename _Scalar> void spdc<_Scalar>::set_sigmas() {
  sigmas_ =
      (0.5 * r::num_ins_ * std::sqrt(r::num_ins_ * r::gamma_ / r::lambda2_)) *
      (prob_ / r::xi_l2norm_);
}

template <typename _Scalar> void spdc<_Scalar>::set_spdc_parameters() {
  const _Scalar n = static_cast<_Scalar>(r::num_ins_);
  const _Scalar tmp = 0.5 * n * std::sqrt(n * r::gamma_ / r::lambda2_);
  set_sigmas();
  tau_ = sigmas_.minCoeff() * (r::num_mbs_ / n);
  theta_ = 1.0 - 1.0 / (1.0 / (r::num_mbs_ * prob_) + tmp / sigmas_).maxCoeff();
}

template <typename _Scalar> void spdc<_Scalar>::set_adaspdc_parameters() {
  const _Scalar n = static_cast<_Scalar>(r::num_ins_);
  const _Scalar tmp = 0.5 * n * std::sqrt(n * r::gamma_ / r::lambda2_);
  _Scalar min_sig = std::numeric_limits<_Scalar>::max();
  _Scalar tmp1 = 0.0;
  for (auto i : selected_dual_coordinates_) {
    min_sig = std::min(min_sig, sigmas_[i]);
    tmp1 = std::max(tmp1, 1.0 / (r::num_mbs_ * prob_[i]) + tmp / sigmas_[i]);
  }
  tau_ = min_sig * (r::num_mbs_ / n);
  theta_ = 1.0 - 1.0 / tmp1;
}
}
#endif
