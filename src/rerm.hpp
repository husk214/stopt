#ifndef STOPT_RERM_HPP
#define STOPT_RERM_HPP

#include <unordered_set>
#include "info.hpp"
#include "io.hpp"

namespace stopt {

enum class regularization_term : int { l1, squared, elastic_net };

enum class loss_term : int {
  smoothed_hinge,
  squared_hinge,
  logistic,
  squared,
  smoothed_insensitve
};

enum class perturbed_algorithm : int { none, adaptreg };

enum class problem_type : int { classification, regression };

template <typename _Scalar, int _Options>
class rerm {
 public:
  rerm(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~rerm() {}

  void set_flag_info(const bool flag);

  void set_parameters(
      const regularization_term r = regularization_term::squared,
      const loss_term l = loss_term::smoothed_hinge, const _Scalar l1 = 1e-3,
      const _Scalar l2 = 1e-3, const _Scalar gamma = 1.0,
      const _Scalar stop_criterion = 1e-6, const uint64_t max_itr = 10000);
  void set_regularization_term(const regularization_term r);
  void set_lambda2(const _Scalar l2);
  void set_stopping_criteria(const _Scalar criteria);
  void set_perturbed_algorithm(const perturbed_algorithm pa);

  _Scalar get_duality_gap();

  _Scalar get_primal_obj_value();
  _Scalar get_regularization_value();
  _Scalar get_loss_value();

  _Scalar get_dual_obj_value();
  _Scalar get_conj_regularization_value();
  _Scalar get_conj_loss_value();

  uint64_t get_total_epoch();

  _Scalar calc_primal_obj_value(const bool flag_cal_margin = true);
  _Scalar calc_regularization_term();
  _Scalar calc_loss_term(const bool flag_cal_margin);

  _Scalar calc_smoothed_hinge_loss(const bool flag_cal_margin);
  _Scalar calc_squared_hinge_loss(const bool flag_cal_margin);
  _Scalar calc_logistic_loss(const bool flag_cal_margin);
  _Scalar calc_squared_loss(const bool flag_cal_margin);
  _Scalar calc_smoothed_insensitive_loss(const bool flag_cal_margin);

  _Scalar calc_duality_gap();

  _Scalar calc_dual_obj_value(const bool flag_cal_yxa_n = true);
  _Scalar calc_conj_regularization_term(const bool flag_cal_yxa_n);
  _Scalar calc_conj_loss_term(
      const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &dv);

  void set_dual_var_kkt();
  void set_primal_var_kkt();
  _Scalar calc_primal_var_sqnorm();

 protected:
  bool check_stopping_criteria();

  _Scalar calc_ip_primal_var_xi(const uint64_t i);
  _Scalar calc_ip_v_xi(const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &v,
                       const uint64_t i);

  void prox_regularized(const _Scalar coeff,
                        const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &ref,
                        const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &g,
                        Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &ans);

  void calc_yxa_n();

  Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> x_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> y_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> margin_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> yxa_n_;

  uint64_t num_ins_;
  uint64_t num_fea_;

  _Scalar lambda1_;  // regularization parameter for l1
  _Scalar lambda2_;  // regularization parameter for squared
  _Scalar lambda_max_;

  _Scalar lambda_;   // regularization function is lambda - strongly convex
  _Scalar gamma_;    // loss function is 1/gamma - smooth
  _Scalar epsilon_;  // epsilon in insensitive or huber loss

  regularization_term regularization_term_;
  loss_term loss_term_;
  problem_type problem_type_;

  _Scalar primal_obj_value_;
  _Scalar regularization_value_;
  _Scalar loss_value_;

  _Scalar dual_obj_value_;
  _Scalar conj_regularization_value_;
  _Scalar conj_loss_value_;

  _Scalar duality_gap_;

  _Scalar stop_criterion_;
  uint64_t max_itr_;
  uint64_t total_epoch_;

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> primal_var_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> dual_var_;
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> xi_l2norm_;
  _Scalar max_xi_norm_;

  uint64_t num_act_ins_;
  uint64_t num_act_fea_;
  std::vector<uint64_t> active_instance_;
  std::vector<uint64_t> active_feature_;

  bool flag_info_;
  bool is_primal_alogrithm_;
  perturbed_algorithm perturbed_algorithm_;
};

template <typename _Scalar, int _Options>
rerm<_Scalar, _Options>::rerm(const std::string &name, const bool &rzv,
                              const bool &flag_info)
    : gamma_(1.0),
      regularization_term_(regularization_term::squared),
      loss_term_(loss_term::smoothed_hinge),
      stop_criterion_(1e-6),
      max_itr_(10000),
      total_epoch_(0),
      flag_info_(flag_info),
      is_primal_alogrithm_(false),
      perturbed_algorithm_(perturbed_algorithm::none) {
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
                                             const _Scalar g, const _Scalar s,
                                             const uint64_t i) {
  regularization_term_ = r;
  loss_term_ = l;
  stop_criterion_ = s;
  max_itr_ = i;
  epsilon_ = 0.1;
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
    gamma_ = 0.25;
    dual_var_.setConstant(0.5);
    primal_var_.setConstant(0.5);
  } else {
    gamma_ = 0.0;
  }

  if (l == loss_term::smoothed_hinge || l == loss_term::squared_hinge ||
      l == loss_term::logistic) {
    problem_type_ = problem_type::classification;
  } else if (l == loss_term::squared || l == loss_term::smoothed_insensitve) {
    problem_type_ = problem_type::regression;
  }
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_regularization_term(
    const regularization_term r) {
  regularization_term_ = r;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_lambda2(const _Scalar l2) {
  lambda2_ = l2;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_stopping_criteria(const _Scalar criteria) {
  stop_criterion_ = criteria;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_perturbed_algorithm(
    const perturbed_algorithm pa) {
  perturbed_algorithm_ = pa;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_duality_gap() {
  return duality_gap_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_primal_obj_value() {
  return primal_obj_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_regularization_value() {
  return regularization_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_loss_value() {
  return loss_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_dual_obj_value() {
  return dual_obj_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_conj_regularization_value() {
  return conj_regularization_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::get_conj_loss_value() {
  return conj_loss_value_;
}

template <typename _Scalar, int _Options>
uint64_t rerm<_Scalar, _Options>::get_total_epoch() {
  return total_epoch_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_primal_obj_value(
    const bool flag_cal_margin) {
  primal_obj_value_ =
      calc_regularization_term() + calc_loss_term(flag_cal_margin);
  return primal_obj_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_regularization_term() {
  regularization_value_ = 0.0;
  switch (regularization_term_) {
    case regularization_term::squared:
      regularization_value_ = 0.5 * lambda2_ * primal_var_.squaredNorm();
      break;
    case regularization_term::l1:
      regularization_value_ = lambda1_ * primal_var_.template lpNorm<1>();
      break;
    case regularization_term::elastic_net:
      regularization_value_ = 0.5 * lambda2_ * primal_var_.squaredNorm() +
                              lambda1_ * primal_var_.template lpNorm<1>();
      break;
  }
  return regularization_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_loss_term(const bool flag_cal_margin) {
  loss_value_ = 0.0;
  switch (loss_term_) {
    case loss_term::smoothed_hinge:
      loss_value_ = calc_smoothed_hinge_loss(flag_cal_margin);
      break;
    case loss_term::squared_hinge:
      loss_value_ = calc_squared_hinge_loss(flag_cal_margin);
      break;
    case loss_term::logistic:
      loss_value_ = calc_logistic_loss(flag_cal_margin);
      break;
    case loss_term::smoothed_insensitve:
      loss_value_ = calc_smoothed_insensitive_loss(flag_cal_margin);
      break;
    case loss_term::squared:
      loss_value_ = calc_squared_loss(flag_cal_margin);
      break;
  }
  return loss_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_smoothed_hinge_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = 1.0 - y_ * (x_ * primal_var_).array();
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
_Scalar rerm<_Scalar, _Options>::calc_squared_hinge_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = 1.0 - y_ * (x_ * primal_var_).array();
  return margin_.cwiseMax(0.0).square().sum() * (0.5 / num_ins_);
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_logistic_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = (-y_ * (x_ * primal_var_).array()).exp();
  return (1.0 + margin_).log().sum() / num_ins_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_smoothed_insensitive_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = (y_ - (x_ * primal_var_).array());
  _Scalar loss = 0.0, tmpi = 0.0;
  const _Scalar tmp1 = 0.5 * gamma_;
  const _Scalar tmp2 = 0.5 / gamma_;
  for (int i = 0; i < num_ins_; ++i) {
    tmpi = std::abs(margin_[i]) - epsilon_;
    if (tmpi > gamma_) {
      loss += tmpi - tmp1;
    } else if (tmpi > 0.0) {
      loss += tmp2 * tmpi * tmpi;
    }
  }
  return loss / num_ins_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_squared_loss(const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = (y_ - (x_ * primal_var_).array());
  return 0.5 * margin_.square().sum() / num_ins_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_duality_gap() {
  duality_gap_ = calc_primal_obj_value();
  if (is_primal_alogrithm_) set_dual_var_kkt();
  duality_gap_ -= calc_dual_obj_value();
  return duality_gap_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_dual_obj_value(
    const bool flag_cal_yxa_n) {
  dual_obj_value_ = -calc_conj_regularization_term(flag_cal_yxa_n) -
                    calc_conj_loss_term(dual_var_);
  return dual_obj_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_conj_regularization_term(
    const bool flag_cal_yxa_n) {
  conj_regularization_value_ = 0.0;
  _Scalar tmp = 0.0;
  switch (regularization_term_) {
    case regularization_term::l1:
      if (flag_cal_yxa_n) calc_yxa_n();
      conj_regularization_value_ = calc_conj_loss_term(
          dual_var_ * (1.0 / (yxa_n_.maxCoeff() / lambda1_)));
      conj_regularization_value_ -= calc_conj_loss_term(dual_var_);
      break;
    case regularization_term::squared:
      if (flag_cal_yxa_n) calc_yxa_n();
      conj_regularization_value_ = (0.5 / lambda2_) * yxa_n_.squaredNorm();
      break;
    case regularization_term::elastic_net:
      if (flag_cal_yxa_n) calc_yxa_n();
      for (auto j : active_feature_) {
        tmp = std::max(0.0, std::abs(yxa_n_.coeff(j)) - lambda1_);
        conj_regularization_value_ += tmp * tmp;
      }
      conj_regularization_value_ *= 0.5 / lambda2_;
      break;
  }
  return conj_regularization_value_;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_conj_loss_term(
    const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &dv) {
  conj_loss_value_ = 0.0;
  switch (loss_term_) {
    case loss_term::smoothed_hinge:
      conj_loss_value_ =
          -(dv.sum() - (0.5 * gamma_) * dv.squaredNorm()) / num_ins_;
      break;
    case loss_term::squared_hinge:
      conj_loss_value_ = -(dv.sum() - 0.5 * dv.squaredNorm()) / num_ins_;
      break;
    case loss_term::logistic:
      conj_loss_value_ = (dv.array() * dv.array().log() +
                          (1.0 - dv.array()) * (1.0 - dv.array()).log())
                             .sum() /
                         num_ins_;
      break;
    case loss_term::smoothed_insensitve:
      conj_loss_value_ =
          (-dv.dot(y_.matrix()) + (0.5 * gamma_) * dv.squaredNorm() +
           epsilon_ * dv.template lpNorm<1>()) /
          num_ins_;
      break;
    case loss_term::squared:
      conj_loss_value_ =
          (0.5 * dv.squaredNorm() - dv.dot(y_.matrix())) / num_ins_;
      break;
  }
  return conj_loss_value_;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::calc_yxa_n() {
  yxa_n_.setZero();
  _Scalar tmp = 0.0;
  const _Scalar one_n = 1.0 / num_ins_;
  if (_Options == Eigen::RowMajor) {
    for (uint64_t i = 0; i < num_ins_; ++i) {
      if (dual_var_[i] != 0.0) {
        switch (problem_type_) {
          case problem_type::classification:
            tmp = y_[i] * dual_var_[i] * one_n;
            break;
          case problem_type::regression:
            tmp = dual_var_[i] * one_n;
            break;
        }
        for (sm_iit<_Scalar, _Options> it(x_, i); it; ++it)
          yxa_n_[it.index()] += tmp * it.value();
      }
    }
  } else if (_Options == Eigen::ColMajor) {
    switch (problem_type_) {
      case problem_type::classification:
        yxa_n_ = x_.transpose() * (one_n * (y_ * dual_var_.array())).matrix();
        break;
      case problem_type::regression:
        yxa_n_ = one_n * x_.transpose() * dual_var_;
        break;
    }
  }
}

template <typename _Scalar, int _Options>
bool rerm<_Scalar, _Options>::check_stopping_criteria() {
  ++total_epoch_;
  calc_duality_gap();
  switch (perturbed_algorithm_) {
    case perturbed_algorithm::none:
      return (duality_gap_ < stop_criterion_) ? true : false;
      break;
    case perturbed_algorithm::adaptreg:
      return (primal_obj_value_ - 0.75 * dual_obj_value_ < stop_criterion_)
                 ? true
                 : false;
      break;
  }
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_ip_primal_var_xi(const uint64_t i) {
  _Scalar value = 0.0;
  for (sm_iit<_Scalar, _Options> it(x_, i); it; ++it)
    value += primal_var_.coeff(it.index()) * it.value();
  return value;
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_ip_v_xi(
    const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &v, const uint64_t i) {
  _Scalar value = 0.0;
  for (sm_iit<_Scalar, _Options> it(x_, i); it; ++it)
    value += v.coeff(it.index()) * it.value();
  return value;
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::prox_regularized(
    const _Scalar coeff, const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &ref,
    const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &g,
    Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &ans) {
  const _Scalar tmp1 = lambda1_ * coeff;
  const _Scalar tmp2 = 1.0 / (lambda2_ * coeff + 1.0);
  _Scalar tmp = 0.0;
  switch (regularization_term_) {
    case regularization_term::l1:
      for (auto j : active_feature_) {
        tmp = ref.coeff(j) - coeff * g.coeff(j);
        if (tmp > tmp1) {
          ans[j] = tmp - tmp1;
        } else if (tmp < -tmp1) {
          ans[j] = tmp + tmp1;
        } else {
          ans[j] = 0.0;
        }
      }
      break;
    case regularization_term::squared:
      ans = (ref - coeff * g) * (1.0 / (1.0 + coeff * lambda2_));
      break;
    case regularization_term::elastic_net:
      for (auto j : active_feature_) {
        tmp = ref.coeff(j) - coeff * g.coeff(j);
        if (tmp > tmp1) {
          ans[j] = (tmp - tmp1) * tmp2;
        } else if (tmp < -tmp1) {
          ans[j] = (tmp + tmp1) * tmp2;
        } else {
          ans[j] = 0.0;
        }
      }
      break;
  }
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_dual_var_kkt() {
  const double one_over_gam = 1.0 / gamma_;
  switch (loss_term_) {
    case loss_term::smoothed_hinge:
      for (int i = 0; i < num_ins_; ++i)
        dual_var_[i] = std::min(1.0, std::max(0.0, one_over_gam * margin_[i]));
      break;
    case loss_term::squared_hinge:
      for (int i = 0; i < num_ins_; ++i)
        dual_var_[i] = std::max(0.0, margin_[i]);
      break;
    case loss_term::logistic:
      for (int i = 0; i < num_ins_; ++i)
        dual_var_[i] =
            std::min(1.0, std::max(0.0, margin_[i] / (margin_[i] + 1.0)));
      break;
    case loss_term::squared:
      dual_var_ = margin_.matrix();
      break;
    case loss_term::smoothed_insensitve:
      for (int i = 0; i < num_ins_; ++i) {
        if (margin_[i] > epsilon_) {
          dual_var_[i] = std::min(1.0, (margin_[i] - epsilon_) / gamma_);
        } else if (margin_[i] < -epsilon_) {
          dual_var_[i] = std::max(-1.0, (margin_[i] + epsilon_) / gamma_);
        } else {
          dual_var_[i] = 0.0;
        }
      }
      break;
  }
}

template <typename _Scalar, int _Options>
void rerm<_Scalar, _Options>::set_primal_var_kkt() {
  const _Scalar one_lambda2 = 1.0 / lambda2_;
  switch (regularization_term_) {
    case regularization_term::squared:
      primal_var_ = one_lambda2 * yxa_n_;
      break;
    case regularization_term::elastic_net:
      for (auto j : active_feature_) {
        primal_var_[j] = sign(yxa_n_[j]) * one_lambda2 *
                         std::max(0.0, std::abs(yxa_n_[j]) - lambda1_);
      }
      break;
    case regularization_term::l1:
      break;
  }
}

template <typename _Scalar, int _Options>
_Scalar rerm<_Scalar, _Options>::calc_primal_var_sqnorm() {
  return primal_var_.squaredNorm();
}
}

#endif
