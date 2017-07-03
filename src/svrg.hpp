#ifndef STOPT_SVRG_HPP
#define STOPT_SVRG_HPP

#include "stochastic.hpp"

namespace stopt {
template <typename _Scalar>
class svrg : public stochastic<_Scalar, Eigen::RowMajor> {
  using r = rerm<_Scalar, Eigen::RowMajor>;
  using s = stochastic<_Scalar, Eigen::RowMajor>;

 public:
  svrg(const std::string &data_libsvm_format,
       const bool &remove_zero_vecotr = true, const bool &flag_info = true);
  ~svrg() {}

  void train();

 protected:
  _Scalar mean_gamma_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> loss_grad_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> vr_loss_grad_;
  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> vr_margin_;

  void calc_full_loss_grad(const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &w);
  void calc_variance_reduced_loss_grad(
      const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &v,
      const bool flag_saga = false);
  void update_variance_reduced_loss_grad(const _Scalar dif, const uint64_t i);
  void update_avg_loss_grad(const _Scalar dif, const uint64_t i);

  void set_prob_data_driven();
};

template <typename _Scalar>
svrg<_Scalar>::svrg(const std::string &name, const bool &rzv,
                    const bool &flag_info)
    : stochastic<_Scalar, Eigen::RowMajor>(name, rzv, flag_info) {
  r::is_primal_alogrithm_ = true;
  mean_gamma_ = r::gamma_ * r::xi_l2norm_.sum() / r::num_ins_;
  loss_grad_.setZero(r::num_fea_);
  vr_loss_grad_.setZero(r::num_fea_);
  vr_margin_.setZero(r::num_fea_);
}

template <typename _Scalar>
void svrg<_Scalar>::train() {
  info<> info_obj{};
  const _Scalar eta = 0.5 / (r::max_xi_norm_ * r::gamma_);
  const int n_a = static_cast<int>(r::num_ins_ / s::num_mbs_);

  for (uint64_t itr = 0; itr < r::max_itr_; ++itr) {
    bool stop = r::check_stopping_criteria();
    if (r::flag_info_)
      info_obj.out_time(itr, r::duality_gap_, r::primal_obj_value_,
                        r::dual_obj_value_);
    if (stop) break;
    calc_full_loss_grad(r::primal_var_);
    for (uint64_t iitr = 0; iitr < n_a; ++iitr) {
      s::sample();
      calc_variance_reduced_loss_grad(r::primal_var_);
      r::prox_regularized(eta, r::primal_var_, vr_loss_grad_, r::primal_var_);
    }
  }
}

template <typename _Scalar>
void svrg<_Scalar>::calc_full_loss_grad(
    const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &w) {
  loss_grad_.setZero();
  _Scalar tmpi = 0.0, tmp = 0.0;
  switch (r::loss_term_) {
    case loss_term::smoothed_hinge:
      vr_margin_ = 1.0 - r::y_ * (r::x_ * w).array();
      for (int i = 0; i < r::num_ins_; ++i) {
        tmpi = -r::y_[i];
        if (vr_margin_[i] > 0.0) {
          if (vr_margin_[i] > r::gamma_) {
            tmpi /= r::num_ins_;
          } else {
            tmpi *= vr_margin_[i] / (r::gamma_ * r::num_ins_);
          }
          for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
            loss_grad_[it.index()] += tmpi * it.value();
        }
      }
      break;
    case loss_term::squared_hinge:
      vr_margin_ = 1.0 - r::y_ * (r::x_ * w).array();
      for (int i = 0; i < r::num_ins_; ++i) {
        if (vr_margin_[i] > 0.0) {
          tmpi = -r::y_[i] * vr_margin_[i] / r::num_ins_;
          for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
            loss_grad_[it.index()] += tmpi * it.value();
        }
      }
      break;
    case loss_term::logistic:
      vr_margin_ = (r::y_ * (r::x_ * w).array()).exp();
      for (int i = 0; i < r::num_ins_; ++i) {
        tmpi = -r::y_[i] / (vr_margin_[i] + 1.0) / r::num_ins_;
        for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
          loss_grad_[it.index()] += tmpi * it.value();
      }
      break;
    case loss_term::squared:
      vr_margin_ = r::y_ - (r::x_ * w).array();
      for (int i = 0; i < r::num_ins_; ++i) {
        tmpi = -vr_margin_[i] / r::num_ins_;
        for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
          loss_grad_[it.index()] += tmpi * it.value();
      }
      break;
    case loss_term::smoothed_insensitve:
      vr_margin_ = r::y_ - (r::x_ * w).array();
      for (int i = 0; i < r::num_ins_; ++i) {
        tmp = std::abs(vr_margin_[i]) - r::epsilon_;
        if (tmp > 0.0) {
          if (tmp > r::gamma_) {
            tmpi = -sign(vr_margin_[i]) / r::num_ins_;
          } else {
            tmpi = -sign(vr_margin_[i]) * tmp / (r::gamma_ * r::num_ins_);
          }
          for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
            loss_grad_[it.index()] += tmpi * it.value();
        }
      }
      break;
  }
}

template <typename _Scalar>
void svrg<_Scalar>::calc_variance_reduced_loss_grad(
    const Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> &v, const bool flag_saga) {
  _Scalar tmp = 0.0, tmp1 = 0.0, tmp2 = 0.0, yi_anpi = 0.0, tmpi = 0.0;
  const _Scalar one_a = 1.0 / s::num_mbs_;
  vr_loss_grad_ = loss_grad_;
  switch (r::loss_term_) {
    case loss_term::smoothed_hinge:
      for (auto i : s::selected_idx_) {
        tmp1 = tmp2 = 0.0;
        yi_anpi = -r::y_[i] * one_a / (r::num_ins_ * s::prob_[i]);
        tmp = 1.0 - r::y_[i] * r::calc_ip_v_xi(v, i);
        if (tmp > r::gamma_) {
          tmp1 = yi_anpi;
        } else if (tmp > 0.0) {
          tmp1 = tmp * yi_anpi / r::gamma_;
        }
        if (vr_margin_[i] > r::gamma_) {
          tmp2 = yi_anpi;
        } else if (vr_margin_[i] > 0.0) {
          tmp2 = yi_anpi * vr_margin_[i] / r::gamma_;
        }
        update_variance_reduced_loss_grad(tmp1 - tmp2, i);
        if (flag_saga) {
          vr_margin_[i] = tmp;
          update_avg_loss_grad((s::num_mbs_ * (tmp1 - tmp2) / r::num_ins_), i);
        }
      }
      break;
    case loss_term::squared_hinge:
      for (auto i : s::selected_idx_) {
        tmp1 = tmp2 = 0.0;
        yi_anpi = -r::y_[i] * one_a / (r::num_ins_ * s::prob_[i]);
        tmp = 1.0 - r::y_[i] * r::calc_ip_v_xi(v, i);
        if (tmp > 0.0) tmp1 = tmp * yi_anpi;
        if (vr_margin_[i] > 0.0) tmp2 = yi_anpi * vr_margin_[i];
        update_variance_reduced_loss_grad(tmp1 - tmp2, i);
        if (flag_saga) {
          vr_margin_[i] = tmp;
          update_avg_loss_grad((s::num_mbs_ * (tmp1 - tmp2) / r::num_ins_), i);
        }
      }
      break;
    case loss_term::logistic:
      for (auto i : s::selected_idx_) {
        tmp1 = tmp2 = 0.0;
        yi_anpi = -r::y_[i] * one_a / (r::num_ins_ * s::prob_[i]);
        tmp = std::exp(r::y_[i] * r::calc_ip_v_xi(v, i));
        tmp1 = yi_anpi / (tmp + 1.0);
        tmp2 = yi_anpi / (vr_margin_[i] + 1.0);
        update_variance_reduced_loss_grad(tmp1 - tmp2, i);
        if (flag_saga) {
          vr_margin_[i] = tmp;
          update_avg_loss_grad((s::num_mbs_ * (tmp1 - tmp2) / r::num_ins_), i);
        }
      }
      break;
    case loss_term::squared:
      for (auto i : s::selected_idx_) {
        tmp1 = tmp2 = 0.0;
        yi_anpi = -one_a / (r::num_ins_ * s::prob_[i]);
        tmp = r::y_[i] - r::calc_ip_v_xi(v, i);
        tmp1 = yi_anpi * tmp;
        tmp2 = yi_anpi * vr_margin_[i];
        update_variance_reduced_loss_grad(tmp1 - tmp2, i);
        if (flag_saga) {
          vr_margin_[i] = tmp;
          update_avg_loss_grad((s::num_mbs_ * (tmp1 - tmp2) / r::num_ins_), i);
        }
      }
      break;
    case loss_term::smoothed_insensitve:
      for (auto i : s::selected_idx_) {
        tmp1 = tmp2 = 0.0;
        yi_anpi = one_a / (r::num_ins_ * s::prob_[i]);
        tmp = r::y_[i] - r::calc_ip_v_xi(v, i);
        tmpi = std::abs(tmp) - r::epsilon_;
        if (tmpi > r::gamma_) {
          tmp1 = -sign(tmp) * yi_anpi;
        } else if (tmpi > 0.0) {
          tmp1 = -sign(tmp) * tmpi * yi_anpi / r::gamma_;
        }
        tmpi = std::abs(vr_margin_[i]) - r::epsilon_;
        if (tmpi > r::gamma_) {
          tmp2 = -sign(vr_margin_[i]) * yi_anpi;
        } else if (tmpi > 0.0) {
          tmp2 = -sign(vr_margin_[i]) * yi_anpi * tmpi / r::gamma_;
        }
        update_variance_reduced_loss_grad(tmp1 - tmp2, i);
        if (flag_saga) {
          vr_margin_[i] = tmp;
          update_avg_loss_grad((s::num_mbs_ * (tmp1 - tmp2) / r::num_ins_), i);
        }
      }
      break;
  }
}

template <typename _Scalar>
void svrg<_Scalar>::update_variance_reduced_loss_grad(const _Scalar dif,
                                                      const uint64_t i) {
  if (!compare(dif, 0.0, 1e-12))
    for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
      vr_loss_grad_[it.index()] += dif * it.value();
}

template <typename _Scalar>
void svrg<_Scalar>::update_avg_loss_grad(const _Scalar dif, const uint64_t i) {
  if (!compare(dif, 0.0, 1e-12))
    for (sm_iit<_Scalar, Eigen::RowMajor> it(r::x_, i); it; ++it)
      loss_grad_[it.index()] += dif * it.value();
}

template <typename _Scalar>
void svrg<_Scalar>::set_prob_data_driven() {
  _Scalar tmp = 1.0 / r::xi_l2norm_.sum();
  s::prob_ = tmp * r::xi_l2norm_;
  std::discrete_distribution<> dst(s::prob_.data(),
                                   s::prob_.data() + s::prob_.size());
  s::sampling_dst_ = std::move(dst);
}
}
#endif
