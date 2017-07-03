#ifndef STOPT_STOCHASTIC_HPP
#define STOPT_STOCHASTIC_HPP

#include "rerm.hpp"

namespace stopt {

enum class sampling : int { uniform, data_driven, optimality_violation };

template <typename _Scalar, int _Options>
class stochastic : public rerm<_Scalar, _Options> {
  using r = rerm<_Scalar, _Options>;

 public:
  stochastic(const std::string &data_libsvm_format,
             const bool &remove_zero_vecotr = true,
             const bool &flag_info = true);
  ~stochastic() {}

  void sample();
  void set_minibatch_size(const uint64_t m);

 protected:
  void prox_loss_conj(const _Scalar &coeff,
                      const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &w,
                      const uint64_t &i);
  _Scalar prox_logistic_conj_newton(const _Scalar &coeff, const _Scalar &yixiw,
                                    const uint64_t &i);
  void update_yxa_n(const _Scalar &dif, const uint64_t &i);

  uint64_t num_mbs_;

  Eigen::Array<_Scalar, Eigen::Dynamic, 1> prob_;
  std::unordered_set<uint64_t> selected_idx_;

  std::mt19937 generator_;
  std::discrete_distribution<> sampling_dst_;
};

template <typename _Scalar, int _Options>
stochastic<_Scalar, _Options>::stochastic(const std::string &name,
                                          const bool &rzv,
                                          const bool &flag_info)
    : rerm<_Scalar, Eigen::RowMajor>(name, rzv, flag_info), num_mbs_(4) {
  prob_.setConstant(r::num_ins_, 1.0 / r::num_ins_);
  std::random_device rd;
  std::mt19937 g(rd());
  generator_ = std::move(g);
  std::discrete_distribution<> dst(prob_.data(), prob_.data() + prob_.size());
  sampling_dst_ = std::move(dst);
  selected_idx_.reserve(num_mbs_);
}

template <typename _Scalar, int _Options>
void stochastic<_Scalar, _Options>::sample() {
  selected_idx_.clear();
  for (int count = 0; count < num_mbs_; ++count)
    selected_idx_.insert(sampling_dst_(generator_));
}

template <typename _Scalar, int _Options>
void stochastic<_Scalar, _Options>::set_minibatch_size(const uint64_t m) {
  num_mbs_ = m;
  selected_idx_.reserve(num_mbs_);
}

template <typename _Scalar, int _Options>
void stochastic<_Scalar, _Options>::prox_loss_conj(
    const _Scalar &s, const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &w,
    const uint64_t &i) {
  _Scalar ai = 0.0, tmp = 0.0, tmp1 = 0.0;
  switch (r::loss_term_) {
    case loss_term::smoothed_hinge:
      tmp = 1.0 / (r::gamma_ * s + 1.0);
      r::margin_[i] = 1.0 - r::y_[i] * r::calc_ip_v_xi(w, i);
      ai = tmp * (s * r::margin_[i] + r::dual_var_[i]);
      ai = std::min(1.0, std::max(0.0, ai));
      update_yxa_n(r::y_[i] * (ai - r::dual_var_[i]) / r::num_ins_, i);
      break;
    case loss_term::squared_hinge:
      tmp = 1.0 / (s + 1.0);
      r::margin_[i] = 1.0 - r::y_[i] * r::calc_ip_v_xi(w, i);
      ai = tmp * (s * r::margin_[i] + r::dual_var_[i]);
      ai = std::max(0.0, ai);
      update_yxa_n(r::y_[i] * (ai - r::dual_var_[i]) / r::num_ins_, i);
      break;
    case loss_term::logistic:
      tmp = r::y_[i] * r::calc_ip_v_xi(w, i);
      // tmp = 1.0 / std::max(1.0, r::gamma_ * s + 1.0);
      // tmp1 = 1.0 / (1.0 + std::exp(r::y_[i] * r::calc_ip_v_xi(w, i)));
      // // ai =  std::min(1.0, std::max(0.0, r::dual_var_[i] + tmp * (s * tmp1
      // -
      // // r::dual_var_[i])));
      // ai = r::dual_var_[i] + tmp * (s * tmp1 - r::y_[i] * r::dual_var_[i]);
      ai = std::min(1.0 - 1e-6,
                    std::max(0.0 + 1e+6, prox_logistic_conj_newton(s, tmp, i)));
      update_yxa_n(r::y_[i] * (ai - r::dual_var_[i]) / r::num_ins_, i);
      break;
    case loss_term::squared:
      tmp = 1.0 / (s + 1.0);
      r::margin_[i] = r::y_[i] - r::calc_ip_v_xi(w, i);
      ai = tmp * (s * r::margin_[i] + r::dual_var_[i]);
      update_yxa_n((ai - r::dual_var_[i]) / r::num_ins_, i);
      break;
    case loss_term::smoothed_insensitve:
      tmp = 1.0 / (r::gamma_ * s + 1.0);
      tmp1 = s * r::epsilon_;
      r::margin_[i] = r::y_[i] - r::calc_ip_v_xi(w, i);
      ai = (s * r::margin_[i] + r::dual_var_[i]);
      if (ai > tmp1) {
        ai = std::min(1.0, std::max(0.0, tmp * (ai - tmp1)));
      } else if (ai < -tmp1) {
        ai = std::max(-1.0, std::min(0.0, tmp * (ai + tmp1)));
      } else {
        ai = 0.0;
      }
      update_yxa_n((ai - r::dual_var_[i]) / r::num_ins_, i);
      break;
  }
  r::dual_var_[i] = ai;
}

template <typename _Scalar, int _Options>
void stochastic<_Scalar, _Options>::update_yxa_n(const _Scalar &dif,
                                                 const uint64_t &i) {
  if (dif != 0.0)
    for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
      r::yxa_n_[it.index()] += dif * it.value();
}

template <typename _Scalar, int _Options>
_Scalar stochastic<_Scalar, _Options>::prox_logistic_conj_newton(
    const _Scalar &coeff, const _Scalar &yixiw, const uint64_t &i) {
  const _Scalar di = r::dual_var_[i];
  const _Scalar one_s = 1.0 / coeff;
  auto f = [&di, &one_s, &yixiw](_Scalar x) {
    return x -
           (yixiw + std::log(x) - std::log(1 - x) + one_s * (x - di)) /
               (one_s + 1.0 / (x - x * x));
  };

  _Scalar pre = r::dual_var_[i];
  _Scalar ans = f(pre);
  while (std::abs(pre - ans) > 1e-6) {
    pre = ans;
    ans = f(pre);
  }
  return ans;
}
}

#endif
