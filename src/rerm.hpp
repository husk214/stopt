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

enum class perturbed_algorithm : int { none, adaptreg, catalyst };

enum class problem_type : int { classification, regression };

template <typename Scalar, int Options>
class rerm {
 public:
  rerm(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~rerm() {}

  void set_flag_info(const bool flag);

  void set_parameters(
      const regularization_term r = regularization_term::squared,
      const loss_term l = loss_term::smoothed_hinge, const Scalar l1 = 1e-3,
      const Scalar l2 = 1e-3, const Scalar gamma = 1.0,
      const Scalar stop_criterion = 1e-6, const uint64_t max_itr = 10000);
  void set_regularization_term(const regularization_term r);
  void set_lambda2(const Scalar l2);
  void set_stopping_criteria(const Scalar criteria);
  void set_perturbed_algorithm(const perturbed_algorithm pa);
  void set_catalyst_kappa(const Scalar kappa);
  void set_catalyst_ref(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ref);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> get_primal_var();

  regularization_term get_regularization_type();
  uint64_t get_num_ins();
  uint64_t get_num_fea();
  Scalar get_lambda1();
  Scalar get_lambda2();
  Scalar get_gamma();

  Scalar get_duality_gap();

  Scalar get_primal_obj_value();
  Scalar get_regularization_value();
  Scalar get_loss_value();

  Scalar get_dual_obj_value();
  Scalar get_conj_regularization_value();
  Scalar get_conj_loss_value();

  uint64_t get_total_epoch();
  Scalar get_stopping_criteria();

  Scalar calc_primal_obj_value(const bool flag_cal_margin = true);
  Scalar calc_regularization_term();
  Scalar calc_regularization_term_catalyst();
  Scalar calc_loss_term(const bool flag_cal_margin);

  Scalar calc_smoothed_hinge_loss(const bool flag_cal_margin);
  Scalar calc_squared_hinge_loss(const bool flag_cal_margin);
  Scalar calc_logistic_loss(const bool flag_cal_margin);
  Scalar calc_squared_loss(const bool flag_cal_margin);
  Scalar calc_smoothed_insensitive_loss(const bool flag_cal_margin);

  Scalar calc_duality_gap();

  Scalar calc_dual_obj_value(const bool flag_cal_yxa_n = true);
  Scalar calc_conj_regularization_term(const bool flag_cal_yxa_n);
  Scalar calc_conj_regularization_term_catalyst(const bool flag_cal_yxa_n);
  Scalar calc_conj_loss_term(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &dv);

  void set_dual_var_kkt();
  void set_primal_var_kkt();
  Scalar calc_primal_var_sqnorm();

 protected:
  bool check_stopping_criteria();

  Scalar calc_ip_primal_var_xi(const uint64_t i);
  Scalar calc_ip_v_xi(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &v,
                      const uint64_t i);

  void prox_regularized(const Scalar coeff,
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ref,
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &g,
                        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ans);

  void prox_regularized_catalyst(
      const Scalar coeff, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ref,
      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &g,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ans);

  void calc_yxa_n();

  Eigen::SparseMatrix<Scalar, Options, std::ptrdiff_t> x_;
  Eigen::Array<Scalar, Eigen::Dynamic, 1> y_;
  Eigen::Array<Scalar, Eigen::Dynamic, 1> margin_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> yxa_n_;

  uint64_t num_ins_;
  uint64_t num_fea_;

  Scalar lambda1_;  // regularization parameter for l1
  Scalar lambda2_;  // regularization parameter for squared
  Scalar lambda_max_;

  Scalar lambda_;   // regularization function is lambda - strongly convex
  Scalar gamma_;    // loss function is 1/gamma - smooth
  Scalar epsilon_;  // epsilon in insensitive or huber loss

  regularization_term regularization_term_;
  loss_term loss_term_;
  problem_type problem_type_;

  Scalar primal_obj_value_;
  Scalar regularization_value_;
  Scalar loss_value_;

  Scalar dual_obj_value_;
  Scalar conj_regularization_value_;
  Scalar conj_loss_value_;

  Scalar duality_gap_;

  Scalar stop_criterion_;
  uint64_t max_itr_;
  uint64_t total_epoch_;

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_var_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_var_;
  Eigen::Array<Scalar, Eigen::Dynamic, 1> xi_l2norm_;
  Scalar max_xi_norm_;

  uint64_t num_act_ins_;
  uint64_t num_act_fea_;
  std::vector<uint64_t> active_instance_;
  std::vector<uint64_t> active_feature_;

  bool flag_info_;
  bool is_primal_alogrithm_;
  perturbed_algorithm perturbed_algorithm_;
  Scalar catalyst_kappa_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> catalyst_ref_;
};

template <typename Scalar, int Options>
rerm<Scalar, Options>::rerm(const std::string &name, const bool &rzv,
                            const bool &flag_info)
    : gamma_(1.0),
      regularization_term_(regularization_term::squared),
      loss_term_(loss_term::smoothed_hinge),
      stop_criterion_(1e-6),
      max_itr_(10000),
      total_epoch_(0),
      flag_info_(flag_info),
      is_primal_alogrithm_(false),
      perturbed_algorithm_(perturbed_algorithm::none),
      catalyst_kappa_(0.0) {
  if (!load_libsvm(x_, y_, name, rzv)) {
    std::cerr << "loading dataset is failed" << std::endl;
  }

  num_ins_ = static_cast<uint64_t>(x_.rows());
  num_fea_ = static_cast<uint64_t>(x_.cols());
  Scalar tmp = 0.0;
  lambda_max_ = 0.0;
  xi_l2norm_.resize(num_ins_);
  xi_l2norm_.setZero();
  max_xi_norm_ = 0.0;
  if (Options == Eigen::RowMajor) {
    std::vector<Scalar> xj_sum(num_fea_, 0.0);
    for (uint64_t i = 0; i < num_ins_; ++i) {
      for (sm_iit<Scalar, Options> it(x_, i); it; ++it) {
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
  } else if (Options == Eigen::ColMajor) {
    for (uint64_t j = 0; j < num_fea_; ++j) {
      Scalar xj_sum = 0.0;
      for (sm_iit<Scalar, Options> it(x_, j); it; ++it) {
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
  catalyst_ref_.setZero(num_fea_);

  margin_.setZero(num_ins_);
  yxa_n_.setZero(num_fea_);

  num_act_ins_ = num_ins_;
  active_instance_.resize(num_act_ins_);
  std::iota(std::begin(active_instance_), std::end(active_instance_), 0);

  num_act_fea_ = num_fea_;
  active_feature_.resize(num_act_fea_);
  std::iota(std::begin(active_feature_), std::end(active_feature_), 0);
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_flag_info(const bool flag) {
  flag_info_ = flag;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_parameters(const regularization_term r,
                                           const loss_term l, const Scalar l1,
                                           const Scalar l2, const Scalar g,
                                           const Scalar s, const uint64_t i) {
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

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_regularization_term(
    const regularization_term r) {
  regularization_term_ = r;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_lambda2(const Scalar l2) {
  lambda2_ = l2;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_stopping_criteria(const Scalar criteria) {
  stop_criterion_ = criteria;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_perturbed_algorithm(
    const perturbed_algorithm pa) {
  perturbed_algorithm_ = pa;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_catalyst_kappa(const Scalar kappa) {
  catalyst_kappa_ = kappa;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_catalyst_ref(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ref) {
  catalyst_ref_ = ref;
}

template <typename Scalar, int Options>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
rerm<Scalar, Options>::get_primal_var() {
  return primal_var_;
}

template <typename Scalar, int Options>
regularization_term rerm<Scalar, Options>::get_regularization_type() {
  return regularization_term_;
}

template <typename Scalar, int Options>
uint64_t rerm<Scalar, Options>::get_num_ins() {
  return num_ins_;
}

template <typename Scalar, int Options>
uint64_t rerm<Scalar, Options>::get_num_fea() {
  return num_fea_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_lambda1() {
  return lambda1_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_lambda2() {
  return lambda2_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_gamma() {
  return gamma_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_duality_gap() {
  return duality_gap_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_primal_obj_value() {
  return primal_obj_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_regularization_value() {
  return regularization_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_loss_value() {
  return loss_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_dual_obj_value() {
  return dual_obj_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_conj_regularization_value() {
  return conj_regularization_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_conj_loss_value() {
  return conj_loss_value_;
}

template <typename Scalar, int Options>
uint64_t rerm<Scalar, Options>::get_total_epoch() {
  return total_epoch_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::get_stopping_criteria() {
  return stop_criterion_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_primal_obj_value(
    const bool flag_cal_margin) {
  primal_obj_value_ =
      calc_regularization_term() + calc_loss_term(flag_cal_margin);
  return primal_obj_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_regularization_term() {
  if (perturbed_algorithm_ == perturbed_algorithm::catalyst) {
    return calc_regularization_term_catalyst();
  } else {
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
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_regularization_term_catalyst() {
  regularization_value_ = 0.0;
  switch (regularization_term_) {
    case regularization_term::squared:
      regularization_value_ =
          0.5 * (lambda2_ + catalyst_kappa_) * primal_var_.squaredNorm();
      break;
    case regularization_term::l1:
      regularization_value_ = lambda1_ * primal_var_.template lpNorm<1>() +
                              0.5 * catalyst_kappa_ * primal_var_.squaredNorm();
      break;
    case regularization_term::elastic_net:
      regularization_value_ =
          0.5 * (lambda2_ + catalyst_kappa_) * primal_var_.squaredNorm() +
          lambda1_ * primal_var_.template lpNorm<1>();
      break;
  }
  regularization_value_ -= catalyst_kappa_ * (primal_var_.dot(catalyst_ref_));
  return regularization_value_;
         
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_loss_term(const bool flag_cal_margin) {
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

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_smoothed_hinge_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = 1.0 - y_ * (x_ * primal_var_).array();
  Scalar sum_loss = 0.0, tmp = 0.0;
  const Scalar p5g = 0.5 * gamma_, p5og = 0.5 / gamma_;
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

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_squared_hinge_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = 1.0 - y_ * (x_ * primal_var_).array();
  return margin_.cwiseMax(0.0).square().sum() * (0.5 / num_ins_);
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_logistic_loss(const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = (-y_ * (x_ * primal_var_).array()).exp();
  return (1.0 + margin_).log().sum() / num_ins_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_smoothed_insensitive_loss(
    const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = (y_ - (x_ * primal_var_).array());
  Scalar loss = 0.0, tmpi = 0.0;
  const Scalar tmp1 = 0.5 * gamma_;
  const Scalar tmp2 = 0.5 / gamma_;
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

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_squared_loss(const bool flag_cal_margin) {
  if (flag_cal_margin) margin_ = (y_ - (x_ * primal_var_).array());
  return 0.5 * margin_.square().sum() / num_ins_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_duality_gap() {
  duality_gap_ = calc_primal_obj_value();
  if (is_primal_alogrithm_) set_dual_var_kkt();
  duality_gap_ -= calc_dual_obj_value();
  return duality_gap_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_dual_obj_value(const bool flag_cal_yxa_n) {
  dual_obj_value_ = -calc_conj_regularization_term(flag_cal_yxa_n) -
                    calc_conj_loss_term(dual_var_);
  return dual_obj_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_conj_regularization_term(
    const bool flag_cal_yxa_n) {
  if (perturbed_algorithm_ == perturbed_algorithm::catalyst) {
    return calc_conj_regularization_term_catalyst(flag_cal_yxa_n);
  } else {
    conj_regularization_value_ = 0.0;
    Scalar tmp = 0.0;
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
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_conj_regularization_term_catalyst(
    const bool flag_cal_yxa_n) {
  conj_regularization_value_ = 0.0;
  Scalar tmp = 0.0;
  switch (regularization_term_) {
    case regularization_term::l1:
      if (flag_cal_yxa_n) calc_yxa_n();
      for (auto j : active_feature_) {
        tmp = std::max(0.0, -lambda1_ + std::abs(yxa_n_.coeff(j) +
                                                 catalyst_kappa_ *
                                                     catalyst_ref_.coeff(j)));        
        conj_regularization_value_ += tmp * tmp;
      }
      conj_regularization_value_ *= 0.5 / catalyst_kappa_;      
      break;
    case regularization_term::squared:
      if (flag_cal_yxa_n) calc_yxa_n();
      conj_regularization_value_ =
          (0.5 / (lambda2_ + catalyst_kappa_)) *
          (yxa_n_ + catalyst_kappa_ * catalyst_ref_).squaredNorm();
      break;
    case regularization_term::elastic_net:
      if (flag_cal_yxa_n) calc_yxa_n();
      for (auto j : active_feature_) {
        tmp = std::max(0.0, -lambda1_ + std::abs(yxa_n_.coeff(j) +
                                                 catalyst_kappa_ *
                                                     catalyst_ref_.coeff(j)));
        conj_regularization_value_ += tmp * tmp;
      }
      conj_regularization_value_ *= 0.5 / (lambda2_ + catalyst_kappa_);
      break;
  }
  return conj_regularization_value_;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_conj_loss_term(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &dv) {
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

template <typename Scalar, int Options>
void rerm<Scalar, Options>::calc_yxa_n() {
  yxa_n_.setZero();
  Scalar tmp = 0.0;
  const Scalar one_n = 1.0 / num_ins_;
  if (Options == Eigen::RowMajor) {
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
        for (sm_iit<Scalar, Options> it(x_, i); it; ++it)
          yxa_n_[it.index()] += tmp * it.value();
      }
    }
  } else if (Options == Eigen::ColMajor) {
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

template <typename Scalar, int Options>
bool rerm<Scalar, Options>::check_stopping_criteria() {
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
    case perturbed_algorithm::catalyst:
      return (primal_obj_value_ - std::max(0.0, dual_obj_value_) <
              stop_criterion_)
                 ? true
                 : false;
      break;
  }
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_ip_primal_var_xi(const uint64_t i) {
  Scalar value = 0.0;
  for (sm_iit<Scalar, Options> it(x_, i); it; ++it)
    value += primal_var_.coeff(it.index()) * it.value();
  return value;
}

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_ip_v_xi(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &v, const uint64_t i) {
  Scalar value = 0.0;
  for (sm_iit<Scalar, Options> it(x_, i); it; ++it)
    value += v.coeff(it.index()) * it.value();
  return value;
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::prox_regularized(
    const Scalar coeff, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ref,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &g,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ans) {
  if (perturbed_algorithm_ == perturbed_algorithm::catalyst) {
    prox_regularized_catalyst(coeff, ref, g, ans);
  } else {
    const Scalar tmp1 = lambda1_ * coeff;
    const Scalar tmp2 = 1.0 / (lambda2_ * coeff + 1.0);
    Scalar tmp = 0.0;
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
}

template <typename Scalar, int Options>
void rerm<Scalar, Options>::prox_regularized_catalyst(
    const Scalar coeff, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ref,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &g,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &ans) {
  const Scalar tmp1 = lambda1_ * coeff;
  const Scalar tmp2 = 1.0 / (lambda2_ * coeff + 1.0 + coeff * catalyst_kappa_);
  Scalar tmp = 0.0;
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
      ans = (ref - coeff * (g - catalyst_kappa_ * catalyst_ref_)) *
            (1.0 / (1.0 + coeff * (lambda2_ + catalyst_kappa_)));
      break;
    case regularization_term::elastic_net:
      for (auto j : active_feature_) {
        tmp = ref.coeff(j) - coeff * g.coeff(j) +
              coeff * catalyst_kappa_ * catalyst_ref_.coeff(j);
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

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_dual_var_kkt() {
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

template <typename Scalar, int Options>
void rerm<Scalar, Options>::set_primal_var_kkt() {
  const Scalar one_lambda2 = 1.0 / lambda2_;
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

template <typename Scalar, int Options>
Scalar rerm<Scalar, Options>::calc_primal_var_sqnorm() {
  return primal_var_.squaredNorm();
}
}  // namespace stopt

#endif
