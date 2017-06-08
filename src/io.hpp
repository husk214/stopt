#ifndef SDM_IO_HPP_
#define SDM_IO_HPP_

#include "utils.hpp"
#include <chrono>
#include <cstdarg>

namespace stopt {

// for readline (used in load libsvm)
static char *sdm_line = nullptr;
static int sdm_max_line_len;
static inline char *readline(FILE *input) {
  if (fgets(sdm_line, sdm_max_line_len, input) == nullptr)
    return nullptr;

  while (strrchr(sdm_line, '\n') == nullptr) {
    sdm_max_line_len *= 2;
    sdm_line = (char *)realloc(sdm_line, sdm_max_line_len);
    int len = (int)strlen(sdm_line);
    if (fgets(sdm_line + len, sdm_max_line_len - len, input) == nullptr)
      break;
  }

  return sdm_line;
}

//////////////////////////////////////////
//             File IO
//////////////////////////////////////////
template <typename Scalar>
bool save(const std::vector<Scalar> v, const std::string &file_name,
          const std::string &separater = " ", const std::string &header = "",
          const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save" << std::endl;
    return false;
  }
  if (header != "")
    output_file << header << std::endl;
  if (size_info)
    output_file << "# " << v.size() << std::endl;

  if (v.empty()) {
    std::cerr << "input file is empty" << std::endl;
    return false;
  } else if (v.size() == 1) {
    output_file << v[0] << std::endl;
  } else {
    for (int i = 0; i < v.size() - 1; ++i)
      output_file << v[i] << separater;
    output_file << v[v.size() - 1] << std::endl;
  }

  return true;
}

template <typename ValueType, int _Rows, int _Cols, int _Major>
bool save(const Eigen::Matrix<ValueType, _Rows, _Cols, _Major> &mat,
          const std::string &file_name, const std::string &separater = " ",
          const std::string &header = "", const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save" << std::endl;
    return false;
  }
  if (header != "")
    output_file << header << std::endl;
  if (size_info) {
    output_file << "# " << mat.rows();
    if (mat.cols() != 1)
      output_file << " " << mat.cols();
    output_file << std::endl;
  }
  if (_Major == Eigen::RowMajor) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        output_file << mat.coeffRef(i, j);
        if (j < mat.cols() - 1)
          output_file << separater;
      }
      output_file << std::endl;
    }
  } else {
    Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rm(mat);
    for (int i = 0; i < rm.rows(); ++i) {
      for (int j = 0; j < rm.cols(); ++j) {
        output_file << rm.coeffRef(i, j);
        if (j < rm.cols() - 1)
          output_file << separater;
      }
      output_file << std::endl;
    }
  }

  return true;
}

template <typename ValueType, int Major, typename Index>
bool save(const Eigen::SparseMatrix<ValueType, Major, Index> &spa_mat,
          const std::string &file_name, const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save\n";
    return false;
  }
  if (size_info)
    output_file << "# " << spa_mat.rows() << " " << spa_mat.cols() << std::endl;

  using InnerIterator = typename Eigen::SparseMatrix<ValueType, Eigen::RowMajor,
                                                     Index>::InnerIterator;
  bool line_break_flag;
  for (int i = 0; i < spa_mat.rows(); ++i) {
    line_break_flag = false;
    for (InnerIterator it(spa_mat, i); it;) {
      output_file << it.index() << ":" << it.value();
      ++it;
      if (it)
        output_file << " ";
      line_break_flag = true;
    }
    if (line_break_flag)
      output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int _Cols, int _Major, typename Index,
          typename _Scalar>
bool save_libsvm(
    const Eigen::Matrix<ValueType, Eigen::Dynamic, _Cols, _Major> &mat,
    const Eigen::Array<_Scalar, Eigen::Dynamic, 1> &y,
    const std::string &file_name, const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "error: in save_libsvm (cannot open the output_file)\n";
    return false;
  }
  if (mat.rows() != y.rows()) {
    std::cerr << "error: in save_libsvm (mat size isn't equal y size)\n";
  }
  if (size_info)
    output_file << "# " << mat.rows() << " " << mat.cols() << std::endl;

  const std::string space = " ";
  const std::string colon = ":";
  for (int i = 0; i < mat.rows(); ++i) {
    output_file << y.coeffRef(i);
    for (int j = 0; j < mat.cols(); ++j)
      output_file << space << j + 1 << colon << mat.coeffRef(i, j);
    output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int Major, typename Index, typename _Scalar>
bool save_libsvm(const Eigen::SparseMatrix<ValueType, Major, Index> &spa_mat,
                 const Eigen::Array<_Scalar, Eigen::Dynamic, 1> &y,
                 const std::string &file_name, const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "error: in save_libsvm (cannot open the output_file)\n";
    return false;
  }
  if (spa_mat.rows() != y.rows()) {
    std::cerr << "error: in save_libsvm (spa_mat size isn't equal y size)\n";
  }
  if (size_info)
    output_file << "# " << spa_mat.rows() << " " << spa_mat.cols() << std::endl;

  using InnerIterator = typename Eigen::SparseMatrix<ValueType, Eigen::RowMajor,
                                                     Index>::InnerIterator;
  bool line_break_flag;
  const std::string space = " ";
  const std::string colon = ":";
  for (int i = 0; i < spa_mat.rows(); ++i) {
    line_break_flag = false;
    output_file << y.coeffRef(i) << space;
    for (InnerIterator it(spa_mat, i); it; ++it)
      output_file << it.index() + 1 << colon << it.value() << space;

    output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int Major>
bool load(Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic, Major> &mat,
          const std::string &file_name, const std::string separater) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  readline(fp);
  std::string buf = sdm_line;
  std::vector<std::string> vec1;
  vec1 = split_string(buf, separater);
  if (vec1.at(0) != "#") {
    std::vector<ValueType> tmp_vec;
    int n = 1, k = 0;
    while (1) {
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;
      tmp_vec.push_back(naive_atot<ValueType>(val));
      ++k;
    }
    while (readline(fp) != nullptr) {
      while (1) {
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        tmp_vec.push_back(naive_atot<ValueType>(val));
      }
      ++n;
    }
    mat = Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>(&tmp_vec[0], n, k);
  } else if (vec1.size() != 3) {
    std::cerr << "file's style error in load matrix" << std::endl;
    return false;
  } else {
    int x_rows = str2int(vec1.at(1));
    int x_cols = str2int(vec1.at(2));
    mat.resize(x_rows, x_cols);
    int n = 0, k = 0;
    while (readline(fp) != nullptr) {
      strtok(sdm_line, " \t");
      k = 0;
      while (1) {
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        mat.coeffRef(n, k) = naive_atot<ValueType>(val);
        ++k;
      }
      ++n;
    }
  }
  fclose(fp);
  return true;
}

template <typename ValueType, int Rows, int Cols, int Major>
bool load(Eigen::Matrix<ValueType, Rows, Cols, Major> &mat,
          const std::string &file_name, const std::string separater = " ",
          const bool &header = false) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::string buf, label_buf, tmp_st1, tmp_st2;
  if (header)
    std::getline(fs, buf);

  std::vector<std::string> vec1;
  std::vector<ValueType> all;
  int num_ins = 0, num_fea = 0, j = 0;
  for (; std::getline(fs, buf); ++num_ins) {
    vec1 = split_string(buf, separater);
    for (j = 0; j < vec1.size(); ++j)
      all.push_back(std::stod(vec1[j]));
    num_fea = std::max(j, num_fea);
  }
  if (Rows == Eigen::Dynamic && Cols == Eigen::Dynamic) {
    mat.resize(num_ins, num_fea);
    mat = Eigen::Map<Eigen::Matrix<ValueType, Rows, Cols, Major>>(
        &all[0], num_ins, num_fea);
  } else if (Rows == Eigen::Dynamic && Cols <= 1) {
    mat.resize(std::max(num_ins, num_fea));
    mat = Eigen::Map<Eigen::Matrix<ValueType, Rows, Cols, Major>>(&all[0],
                                                                  all.size());

  } else if (Cols == Eigen::Dynamic && Rows <= 1) {
    mat.resize(std::max(num_ins, num_fea));
    mat = Eigen::Map<Eigen::Matrix<ValueType, Rows, Cols, Major>>(&all[0],
                                                                  all.size());
  }
  fs.close();
  return true;
}

template <typename ValueType, int Major, typename Index>
bool load(Eigen::SparseMatrix<ValueType, Major, Index> &spa_mat,
          const std::string &file_name) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<ValueType>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);

  std::string buf;
  int n, d, k;
  n = d = 0;
  std::string::size_type idx0 = static_cast<std::string::size_type>(0);
  std::string::size_type idx1 = idx0, idx2 = idx0;
  double tmp = 0;
  while (std::getline(fs, buf)) {
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cerr << "file format error in load SpaMat" << std::endl;
      return false;
    }
    idx1 = idx0, idx2 = idx0;
    do {
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = std::atoi((buf.substr(idx1, idx2 - idx1)).c_str());
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);
      tmp = naive_atot<ValueType>((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList.push_back(Tri(n, k, tmp));
    } while (idx1 != std::string::npos);
    ++n;
  }
  fs.close();
  ++d;
  spa_mat.resize(n, d);
  spa_mat.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_mat.makeCompressed();
  return true;
}

template <typename _Scalar, int Major, typename Index>
bool load_libsvm(Eigen::SparseMatrix<_Scalar, Major, Index> &spa_x,
                 Eigen::Array<_Scalar, Eigen::Dynamic, 1> &y,
                 const std::string &file_name,
                 const bool &flag_remove_zero = true) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);
  y.resize(1024);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  unsigned int n = 0, d = 0, k = 0, num_ele = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    y[n] = naive_atot<_Scalar>(p);
    num_ele = 0;
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;
      ++num_ele;
      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      tripletList.push_back(Tri(n, k, naive_atot<_Scalar>(val)));
    }
    if (!flag_remove_zero || (flag_remove_zero && num_ele > 0))
      ++n;
    if (static_cast<unsigned int>(y.size()) <= n)
      y.conservativeResize(y.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  y.conservativeResize(n);

  spa_x.resize(n, d);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename _Scalar, int Major>
bool load_libsvm(
    Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Major> &dense_x,
    Eigen::Array<_Scalar, Eigen::Dynamic, 1> &y,
    const std::string &file_name) {

  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::vector<_Scalar> x_vec;
  x_vec.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  _Scalar vt1_0;

  unsigned int n = 0, d = 0, k = 0, pre_k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    y[n] = naive_atot<_Scalar>(p);

    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      for (; pre_k < k; ++pre_k)
        x_vec.push_back(vt1_0);

      x_vec.push_back(naive_atot<_Scalar>(val));
      pre_k = k + 1;
    }

    if (static_cast<unsigned int>(y.size()) <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  y.conservativeResize(n);
  dense_x = Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(&x_vec[0], n, k + 1);
  return true;
}

template <typename _Scalar, int Major, typename Index>
bool load_libsvm_subsampling(
    Eigen::SparseMatrix<_Scalar, Major, Index> &spa_x,
    Eigen::Array<_Scalar, Eigen::Dynamic, 1> &y,
    const std::string &file_name, const int &num_ins, const int &num_fea,
    const bool &random_seed_flag = false) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "error: in load_libsvm_subsampling (file open error)\n";
    return false;
  }
  Eigen::Array<_Scalar, Eigen::Dynamic, 1> whole_y;
  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<Tri> tripletList;
  std::vector<std::vector<Tri>> tri_ins;
  tripletList.reserve(1024);
  whole_y.resize(1024);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  int n = 0, d = 0, k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    whole_y[n] = naive_atot<_Scalar>(p);
    std::vector<Tri> ins_tri_vec;
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      ins_tri_vec.push_back(Tri(n, k, naive_atot<_Scalar>(val)));
    }
    tri_ins.push_back(ins_tri_vec);
    if (static_cast<unsigned int>(whole_y.size()) <= (++n))
      whole_y.conservativeResize(whole_y.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  whole_y.conservativeResize(n);

  int sub_num_ins = std::min(num_ins, n);
  int sub_num_fea = std::min(num_fea, d);
  y.resize(sub_num_ins);

  std::vector<int> ins_index(n);
  std::iota(std::begin(ins_index), std::end(ins_index), 0);

  std::vector<int> fea_index(d);
  std::iota(std::begin(fea_index), std::end(fea_index), 0);

  std::mt19937 engine;
  std::random_device rnd;
  std::vector<std::uint_least32_t> v(10);
  std::generate(std::begin(v), std::end(v), std::ref(rnd));
  std::seed_seq seed(std::begin(v), std::end(v));
  if (random_seed_flag)
    engine.seed(seed);
  std::shuffle(std::begin(ins_index), std::end(ins_index), engine);

  std::generate(std::begin(v), std::end(v), std::ref(rnd));
  if (random_seed_flag)
    engine.seed(seed);
  std::shuffle(std::begin(fea_index), std::end(fea_index), engine);

  Eigen::VectorXi flag_subsample_fea = Eigen::VectorXi::Zero(d);
  Eigen::VectorXi new_fea_index = Eigen::VectorXi::Zero(d);
  for (int j = 0; j < sub_num_fea; ++j)
    new_fea_index[fea_index[j]] = j;

  for (int j = 0; j < sub_num_fea; ++j)
    flag_subsample_fea[fea_index[j]] = 1;

  for (int i = 0; i < sub_num_ins; ++i) {
    y.coeffRef(i) = whole_y.coeffRef(ins_index[i]);
    for (auto &&j : tri_ins[ins_index[i]]) {
      if (flag_subsample_fea[j.col()])
        tripletList.push_back(Tri(i, new_fea_index[j.col()], j.value()));
    }
  }

  spa_x.resize(sub_num_ins, sub_num_fea);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename _Scalar, int Major, typename Index>
bool load_libsvm_binary(Eigen::SparseMatrix<_Scalar, Major, Index> &spa_x,
                        Eigen::Array<_Scalar, Eigen::Dynamic, 1> &label,
                        const std::string &file_name,
                        const bool flag_remove_zero = false) {
  // if flag_remove_zero is true
  // then the instance \|x_i\|_2 = 0 is not included in spa_x
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);

  _Scalar label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  _Scalar label_true, label_false;
  label_true = static_cast<_Scalar>(1.0);
  label_false = static_cast<_Scalar>(-1.0);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));
  label.resize(1024);

  unsigned int n = 0, d = 0, k = 0, each_nnz = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }

    tmp_label = naive_atot<_Scalar>(p);
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      label[n] = ((tmp_label == label_memo) ? 1.0 : -1.0);
    } else {
      if (label_memo == tmp_label) {
        label[n] = label_true;
      } else {
        if (label_memo > tmp_label) {
          label[n] = label_false;
        } else {
          for (unsigned int i = 0; i < n; ++i)
            label.coeffRef(i) = label_false;
          label[n] = label_true;
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    each_nnz = 0;
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;
      ++each_nnz;
      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      tripletList.push_back(Tri(n, k, naive_atot<_Scalar>(val)));
    }
    if (!flag_remove_zero || (flag_remove_zero && each_nnz != 0))
      ++n;

    if (static_cast<unsigned int>(label.size()) <= n)
      label.conservativeResize(label.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  label.conservativeResize(n);

  spa_x.resize(n, d);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename _Scalar, int Major, typename Index>
bool load_libsvm_binary_randomly(
    Eigen::SparseMatrix<_Scalar, Major, Index> &spa_x,
    Eigen::Array<_Scalar, Eigen::Dynamic, 1> &label,
    const std::string &file_name, const bool seed_flag = false) {
  int num_ins = count_lines(file_name);
  std::vector<int> whole_index(num_ins);
  std::iota(std::begin(whole_index), std::end(whole_index), 0);

  std::mt19937 engine;
  if (seed_flag) {
    std::random_device rnd;
    std::vector<std::uint_least32_t> v(10);
    std::generate(std::begin(v), std::end(v), std::ref(rnd));
    std::seed_seq seed(std::begin(v), std::end(v));
    engine.seed(seed);
  }
  std::shuffle(std::begin(whole_index), std::end(whole_index), engine);

  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);

  _Scalar label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  _Scalar label_true, label_false;
  label_true = static_cast<_Scalar>(1.0);
  label_false = static_cast<_Scalar>(-1.0);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));
  label.resize(num_ins);

  unsigned int n = 0, d = 0, k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }

    tmp_label = naive_atot<_Scalar>(p);
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      label.coeffRef(whole_index[n]) = ((tmp_label == label_memo) ? 1.0 : -1.0);
    } else {
      if (label_memo == tmp_label) {
        label.coeffRef(whole_index[n]) = label_true;
      } else {
        if (label_memo > tmp_label) {
          label.coeffRef(whole_index[n]) = label_false;
        } else {
          for (unsigned int i = 0; i < n; ++i)
            label.coeffRef(whole_index[i]) = label_false;
          label.coeffRef(whole_index[n]) = label_true;
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      tripletList.push_back(
          Tri(whole_index[n], k, naive_atot<_Scalar>(val)));
    }
    ++n;
  }
  fclose(fp);
  free(sdm_line);
  ++d;

  spa_x.resize(n, d);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename _Scalar, int Major>
bool load_libsvm_binary(
    Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, Major> &dense_x,
    Eigen::Array<_Scalar, Eigen::Dynamic, 1> &label,
    const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::vector<_Scalar> x_vec;
  x_vec.reserve(1024); // estimation of non_zero_entries
  label.resize(1024);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  _Scalar tmp, vt1_0;
  tmp = vt1_0 = static_cast<_Scalar>(0.0);
  _Scalar label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  _Scalar label_true, label_false;
  label_true = static_cast<_Scalar>(1.0);
  label_false = static_cast<_Scalar>(-1.0);

  unsigned int n = 0, d = 0, k = 0, pre_k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    tmp_label = naive_atot<_Scalar>(p);
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      label[n] = ((tmp_label == label_memo) ? 1.0 : -1.0);
    } else {
      if (label_memo == tmp_label) {
        label[n] = (label_true);
      } else {
        if (label_memo > tmp_label) {
          label[n] = (label_false);
        } else {
          for (int i = 0; i < n; ++i)
            label.coeffRef(i) = -1.0;
          label[n] = (label_true);
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      for (; pre_k < k; ++pre_k)
        x_vec.push_back(vt1_0);

      x_vec.push_back(naive_atot<_Scalar>(val));
      pre_k = k + 1;
    }

    if (label.size() <= (++n))
      label.conservativeResize(label.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  label.conservativeResize(n);
  dense_x = Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(&x_vec[0], n, k + 1);
  return true;
}

// for equal_split_libsvm_binary_for_cross_validation
// - vec_flag control where he present instance should push
static inline int get_to_distribute_index(const std::vector<bool> &vec_flag,
                                          const int &vec_size) {
  for (int i = 0; i < vec_size; ++i) {
    if (vec_flag[i] == true)
      return i;
  }
  std::cerr << "error in get_to_distribute_index" << std::endl;
  return 0;
}

static inline void change_vec_flag(std::vector<bool> &vec_flag,
                                   const int &vec_size, const int &index) {
  vec_flag[index] = false;
  int next_index = index + 1;
  if (next_index == vec_size) {
    vec_flag[0] = true;
  } else {
    vec_flag[next_index] = true;
  }
}

template <typename _Scalar, int Major, typename Index>
bool equal_split_libsvm_binary_for_cross_validation(
    std::vector<Eigen::SparseMatrix<_Scalar, Major, Index>> &vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<_Scalar, Major, Index>> &vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y, const std::string &file_name,
    const int &split_num) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<_Scalar> *tmpy_array = new std::vector<_Scalar>[split_num];
  std::vector<Tri> *triplets_array = new std::vector<Tri>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::vector<int> vec_n(split_num, 0);
  std::vector<bool> vec_flag_p(split_num, false);
  std::vector<bool> vec_flag_n(split_num, false);
  int d, k, l, last_index, the_line_index;
  d = k = l = last_index = the_line_index = 0;
  _Scalar label_memo, label_tmp;
  label_memo = static_cast<_Scalar>(0.0);
  label_tmp = label_memo;
  vec_flag_p.at(0) = true;
  vec_flag_n.at(0) = true;
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (readline(fp) == nullptr) {
        last_index = i;
        goto LABEL;
      }
      char *p = strtok(sdm_line, " \t\n");
      if (p == nullptr) {
        std::cerr << "error: empty line" << std::endl;
        return false;
      }
      label_tmp = naive_atot<_Scalar>(p);
      if (l == 0)
        label_memo = label_tmp;

      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }
      while (1) {
        char *idx = strtok(nullptr, ":");
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        k = strtol(idx, nullptr, 10) - 1;
        if (d < k)
          d = k;
        (triplets_array[the_line_index])
            .push_back(
                Tri(vec_n[the_line_index], k, naive_atot<_Scalar>(val)));
      }
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL:
  fclose(fp);
  free(sdm_line);
  ++d;
  std::vector<Tri> vec_x;
  std::vector<_Scalar> vec_y;

  for (int i = 0; i < split_num; ++i) {
    vec_train_x[i].resize(l - vec_n[i], d);
    vec_valid_x[i].resize(vec_n[i], d);

    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(Tri((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i])
        .setFromTriplets((triplets_array[i]).begin(),
                         (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
  return true;
}

template <typename _Scalar, int Major, typename Index>
bool merge_equal_split_libsvm_binary_for_cross_validation(
    std::vector<Eigen::SparseMatrix<_Scalar, Major, Index>> &vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<_Scalar, Major, Index>> &vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y, const std::string &file_name1,
    const std::string &file_name2, const int &split_num) {
  FILE *fp = fopen(file_name1.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error file1" << std::endl;
    return false;
  }
  FILE *fp2 = fopen(file_name2.c_str(), "r");
  if (fp2 == nullptr) {
    std::cerr << "file open error file2" << std::endl;
    return false;
  }
  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<_Scalar> *tmpy_array = new std::vector<_Scalar>[split_num];
  std::vector<Tri> *triplets_array = new std::vector<Tri>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::vector<int> vec_n(split_num, 0);
  std::vector<bool> vec_flag_p(split_num, false);
  std::vector<bool> vec_flag_n(split_num, false);
  int d, k, l, last_index, the_line_index;
  d = k = l = last_index = the_line_index = 0;
  _Scalar label_memo, label_tmp;
  label_memo = static_cast<_Scalar>(0.0);
  label_tmp = label_memo;
  vec_flag_p.at(0) = true;
  vec_flag_n.at(0) = true;
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (readline(fp) == nullptr) {
        last_index = i;
        goto LABEL;
      }
      char *p = strtok(sdm_line, " \t\n");
      if (p == nullptr) {
        std::cerr << "error: empty line" << std::endl;
        return false;
      }
      label_tmp = naive_atot<_Scalar>(p);
      if (l == 0)
        label_memo = label_tmp;

      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }
      while (1) {
        char *idx = strtok(nullptr, ":");
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        k = strtol(idx, nullptr, 10) - 1;
        if (d < k)
          d = k;
        (triplets_array[the_line_index])
            .push_back(
                Tri(vec_n[the_line_index], k, naive_atot<_Scalar>(val)));
      }
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL:
  fclose(fp);
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (readline(fp2) == nullptr) {
        last_index = i;
        goto LABEL2;
      }
      char *p = strtok(sdm_line, " \t\n");
      if (p == nullptr) {
        std::cerr << "error: empty line" << std::endl;
        return false;
      }
      label_tmp = naive_atot<_Scalar>(p);

      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }
      while (1) {
        char *idx = strtok(nullptr, ":");
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        k = strtol(idx, nullptr, 10) - 1;
        if (d < k)
          d = k;
        (triplets_array[the_line_index])
            .push_back(
                Tri(vec_n[the_line_index], k, naive_atot<_Scalar>(val)));
      }
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL2:
  fclose(fp2);

  ++d;
  std::vector<Tri> vec_x;
  std::vector<_Scalar> vec_y;

  for (int i = 0; i < split_num; ++i) {
    vec_train_x[i].resize(l - vec_n[i], d);
    vec_valid_x[i].resize(vec_n[i], d);

    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(Tri((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i])
        .setFromTriplets((triplets_array[i]).begin(),
                         (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  free(sdm_line);
  delete[] triplets_array;
  delete[] tmpy_array;
  return true;
}

// data x is splited two part (x1 and x2).
// the number of x1's samples is (parcentage_x1 * the number of x's samples).
// x2 is the rest of x1.
template <typename _Scalar, int Major, typename Index>
bool split_libsvm_binary(Eigen::SparseMatrix<_Scalar, Major, Index> &x1,
                         Eigen::ArrayXd &y1,
                         Eigen::SparseMatrix<_Scalar, Major, Index> &x2,
                         Eigen::ArrayXd &y2, const std::string &file_name,
                         const double percentage_x1, const bool &flag_random) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));
  int num_ins = 0;
  _Scalar label1, tmp_label;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    if (num_ins == 0) {
      label1 = naive_atot<_Scalar>(p);
    }
    ++num_ins;
  }
  fclose(fp);
  int x1_size = percentage_x1 * num_ins;
  int x2_size = num_ins - x1_size;
  std::vector<int> flag_x1(num_ins, 1);
  for (int i = 0; i < x1_size; ++i)
    flag_x1[i] = 0;
  unsigned seed;
  if (flag_random) {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  } else {
    seed = 42;
  }
  std::shuffle(std::begin(flag_x1), std::end(flag_x1),
               std::default_random_engine(seed));

  using Tri = Eigen::Triplet<_Scalar>;
  std::vector<std::vector<Tri>> tripletLists(2);
  (tripletLists[0]).reserve(x1_size);
  (tripletLists[1]).reserve(x2_size);
  std::vector<Eigen::ArrayXd> ys(2);
  (ys[0]).resize(x1_size);
  (ys[1]).resize(x2_size);
  std::vector<int> ns(2, 0);
  fp = fopen(file_name.c_str(), "r");
  int wh = 0;
  unsigned int d = 0, k = 0;
  for (int index = 0; readline(fp) != nullptr; ++index) {
    char *p = strtok(sdm_line, " \t\n");
    tmp_label = naive_atot<_Scalar>(p);
    wh = flag_x1[index];
    (ys[wh])[ns[wh]] = ((tmp_label == label1) ? 1.0 : -1.0);
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      tripletLists[wh].push_back(Tri(ns[wh], k, naive_atot<_Scalar>(val)));
    }
    ++ns[wh];
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  x1.resize(x1_size, d);
  x2.resize(x2_size, d);
  x1.setFromTriplets(std::begin(tripletLists[0]), std::end(tripletLists[0]));
  x2.setFromTriplets(std::begin(tripletLists[1]), std::end(tripletLists[1]));
  x1.makeCompressed();
  x2.makeCompressed();
  y1 = ys[0];
  y2 = ys[1];
  return true;
}

} // namespace sdm

#endif
