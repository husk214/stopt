#ifndef STOPT_RERM_HPP
#define STOPT_RERM_HPP

#include "info.hpp"
#include "io.hpp"

namespace stopt {

enum class regularization_term : int { squared, l1, elastic_net };

enum class loss_term : int {
  smoothed_hinge,
  squared_hinge,
  logistic,
  squared,
  smoothed_insensitve
};

enum class sampling : int { uniform, data_driven, optimality_violation };

template <typename _Scalar, int _Options> class rerm {
public:
  rerm(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~rerm() {}

  void set_flag_info(const bool flag);

  void
  set_parameters(const regularization_term r = regularization_term::squared,
                 const loss_term l = loss_term::smoothed_hinge,
                 const _Scalar l1 = 1e-3, const _Scalar l2 = 1e-3,
                 const _Scalar gamma = 1.0, _Scalar stop_criterion = 1e-6,
                 uint64_t max_itr = 10000);

  _Scalar calc_primal_obj_value();
  _Scalar calc_regularization_term();
  _Scalar calc_loss_term();

  _Scalar calc_smoothed_hinge_loss();
  _Scalar calc_squared_hinge_loss();
  _Scalar calc_logistic_loss();
  _Scalar zcalc_squared_loss();
  _Scalar calc_smoothed_insensitive_loss();

  _Scalar calc_duality_gap();

  _Scalar calc_dual_obj_value();
  _Scalar calc_conj_regularization_term();
  _Scalar calc_conj_loss_term();
  void calc_yxa_n();

  Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> x_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> y_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> margin_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> yxa_n_;

  uint64_t num_ins_;
  uint64_t num_fea_;
  uint64_t num_mbs_;

  _Scalar lambda1_; // regularization parameter for l1
  _Scalar lambda2_; // regularization parameter for squared
  _Scalar lambda_max_;

  _Scalar lambda_; // regularization function is lambda - strongly convex
  _Scalar gamma_;  // loss function is 1/gamma - smooth

  regularization_term regularization_term_;
  loss_term loss_term_;

  _Scalar stop_criterion_;
  uint64_t max_itr_;

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> primal_var_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> dual_var_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> xi_l2norm_;
  _Scalar max_xi_norm_;

  uint64_t num_act_ins_;
  uint64_t num_act_fea_;
  std::vector<uint64_t> active_instance_;
  std::vector<uint64_t> active_feature_;

  bool flag_info_;
  bool safe_instance_screening_;
  bool safe_feature_screening_;
};

template <typename _Scalar, int _Options>
rerm<_Scalar, _Options>::rerm(const std::string &name, const bool &rzv,
                              const bool &flag_info)
    : num_mbs_(1), gamma_(1.0),
      regularization_term_(regularization_term::squared),
      loss_term_(loss_term::smoothed_hinge), stop_criterion_(1e-6),
      max_itr_(10000), flag_info_(flag_info), safe_instance_screening_(false),
      safe_feature_screening_(false) {
  if (!load_libsvm(x_, y_, name, rzv)) {
    std::cerr << "loading dataset is failed" << std::endl;
  }

  num_ins_ = static_cast<uint64_t>(x_.rows());
  num_fea_ = static_cast<uint64_t>(x_.cols());
  _Scalar tmp = 0.0;
  lambda_max_ = 0.0;
  xi_l2norm_.resize(num_ins_);
  xi_l2norm_.setZero();
  max_xi_norm_ = 0.0;
  if (_Options == Eigen::RowMajor) {
    std::vector<_Scalar> xj_sum(num_fea_, 0.0);
    for (uint64_t i = 0; i < num_ins_; ++i) {
      for (sm_iit<_Scalar, _Options> it(x_, i); it; ++it) {
        tmp = it.value();
        xi_l2norm_[i] += tmp * tmp;
        xj_sum[it.index()] += tmp;
      }
      xi_l2norm_[i] = std::sqrt(xi_l2norm_[i]);
      max_xi_norm_ = std::max(max_xi_norm_, xi_l2norm_[i]);
    }
    for (uint64_t j = 0; j < num_fea_; ++j)
      lambda_max_ = std::max(lambda_max_, xj_sum[j]);
    lambda_max_ /= num_ins_;
  } else if (_Options == Eigen::ColMajor) {
    for (uint64_t j = 0; j < num_fea_; ++j) {
      _Scalar xj_sum = 0.0;
      for (sm_iit<_Scalar, _Options> it(x_, j); it; ++it) {
        tmp = it.value();
        xj_sum += tmp;
        xi_l2norm_[it.index()] += tmp * tmp;
      }
      lambda_max_ = std::max(lambda_max_, xj_sum);
    }
    for (uint64_t i = 0; i < num_ins_; ++i)
      max_xi_norm_ = std::max(max_xi_norm_, xi_l2norm_[i]);
    lambda_max_ /= num_ins_;
  } else {
    lambda_max_ = 1.0 / num_ins_;
  }

  lambda1_ = 1e-3 * lambda_max_;
  lambda2_ = 1e-3 * lambda_max_;

  primal_var_.setZero(num_fea_);
  dual_var_.setZero(num_ins_);

  margin_.setZero(num_ins_);
  yxa_n_.setZero(num_fea_);

  num_act_ins_ = num_ins_;
  active_instance_.resize(num_act_ins_);
  std::iota(std::begin(active_instance_), std::end(active_instance_), 0);

  num_act_fea_ = num_fea_;
  active_feature_.resize(num_act_fea_);
  std::iota(std::begin(active_feature_), std::end(active_feature_), 0);
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_flag_info(const bool flag) {
  flag_info_ = flag;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_parameters(const regularization_term r,
                                             const loss_term l,
                                             const _Scalar l1, const _Scalar l2,
                                             const _Scalar g, _Scalar s,
                                             uint64_t i) {
  regularization_term_ = r;
  loss_term_ = l;
  stop_criterion_ = s;
  max_itr_ = i;

  switch (regularization_term_) {
  case regularization_term::l1:
    lambda1_ = l1;
    lambda2_ = 0.0;
    lambda_ = 0.0;
    break;
  case regularization_term::squared:
    lambda1_ = 0.0;
    lambda2_ = l2;
    lambda_ = lambda2_;
    break;
  case regularization_term::elastic_net:
    lambda1_ = l1;
    lambda2_ = l2;
    lambda_ = lambda2_;
    break;
  }

  if (l == loss_term::smoothed_hinge || l == loss_term::smoothed_insensitve) {
    gamma_ = g;
  } else if (l == loss_term::squared || l == loss_term::squared_hinge) {
    gamma_ = 1.0;
  } else if (l == loss_term::logistic) {
    gamma_ = 4.0;
  } else {
    gamma_ = 0.0;
  }
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_primal_obj_value() {
  return calc_regularization_term() + calc_loss_term();
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_regularization_term() {
  _Scalar value = 0.0;
  switch (regularization_term_) {
  case regularization_term::squared:
    value = 0.5 * lambda2_ * primal_var_.squaredNorm();
    break;
  case regularization_term::l1:
    value = lambda1_ * primal_var_.template lpNorm<1>();
    break;
  case regularization_term::elastic_net:
    value = 0.5 * lambda2_ * primal_var_.squaredNorm() +
            lambda1_ * primal_var_.template lpNorm<1>();
    break;
  }
  return value;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_loss_term() {
  _Scalar value = 0.0;
  switch (loss_term_) {
  case loss_term::smoothed_hinge:
    value = calc_smoothed_hinge_loss();
    break;
  case loss_term::squared_hinge:
    value = calc_squared_hinge_loss();
    break;
  }
  return value;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_smoothed_hinge_loss() {
  margin_ = 1.0 - y_ * (x_ * primal_var_).array();
  _Scalar sum_loss = 0.0, tmp = 0.0;
  const _Scalar p5g = 0.5 * gamma_, p5og = 0.5 / gamma_;
  for (uint64_t i = 0; i < num_ins_; ++i) {
    tmp = margin_[i];
    if (tmp > gamma_) {
      sum_loss += tmp - p5g;
    } else if (tmp >= 0.0) {
      sum_loss += p5og * tmp * tmp;
    }
  }
  return sum_loss / num_ins_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_squared_hinge_loss() {
  margin_ = 1.0 - y_ * (x_ * primal_var_).array();
  return margin_.cwiseMax(0.0).square().sum() * (0.5 / num_ins_);
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_logistic_loss() {
  margin_ = - y_ * (x_ * primal_var_).array();
  return (1.0 +  margin_.exp()).log() / num_ins_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_duality_gap() {
  return calc_primal_obj_value() - calc_dual_obj_value();
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_dual_obj_value() {
  return -calc_conj_regularization_term() - calc_conj_loss_term();
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_conj_regularization_term() {
  _Scalar value = 0.0, tmp = 0.0;
  switch (regularization_term_) {
  case regularization_term::l1:
    break;
  case regularization_term::squared:
    calc_yxa_n();
    value = (0.5 / lambda2_) * yxa_n_.squaredNorm();
    break;
  case regularization_term::elastic_net:
    calc_yxa_n();
    for (auto j : active_feature_) {
      tmp = std::max(0.0, std::abs(yxa_n_.coeff(j)) - lambda1_);
      value += tmp * tmp;
    }
    value *= 0.5 / lambda2_;
    break;
  }
    return value;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_conj_loss_term() {
  _Scalar value = 0.0;
  switch (loss_term_) {
  case loss_term::smoothed_hinge:
    value =
        -(dual_var_.sum() - (0.5 * gamma_) * dual_var_.squaredNorm()) / num_ins_;
    break;
  case loss_term::squared_hinge:
    value = -(dual_var_.sum() - 0.5 * dual_var_.squaredNorm()) / num_ins_;
    break;
  }
  return value;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::calc_yxa_n() {
  yxa_n_.setZero();
  _Scalar tmp = 0.0;
  const _Scalar one_n = 1.0 / num_ins_;
  if (_Options == Eigen::RowMajor) {
    for (uint64_t i = 0; i < num_ins_; ++i) {
      if (dual_var_[i] != 0.0) {
        tmp = y_[i] * dual_var_[i] * one_n;
        for (sm_iit<_Scalar, _Options> it(x_, i); it; ++it)
          yxa_n_[it.index()] += tmp * it.value();
      }
    }
  } else if (_Options == Eigen::ColMajor) {
    yxa_n_ = x_.transpose() * (one_n * (y_ * dual_var_.array())).matrix();
  }
}
}

#endif
