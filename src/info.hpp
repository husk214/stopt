#ifndef STOPT_INFO_HPP
#define STOPT_INFO_HPP

#include "utils.hpp"

namespace stopt {

template <typename _UnitOfTime = std::chrono::milliseconds,
          typename _ClockType = std::chrono::system_clock>
class info {
public:
  explicit info(const std::string &&delim = ",");
  ~info();

  void set_cout_default();
  void set_delim(const std::string &&delim = ",");
  void set_time_format(const std::string &&time_format = "%5.3e");

  void print(const char *fmt, ...);
  void print_time(const char *fmt, ...);

  void out() {}
  template <class Head, class... Tail> void out(Head &&h, Tail &&... t);

  void outnl() {}
  template <class Head, class... Tail> void outnl(Head &&h, Tail &&... t);

  template <class... All> void out_time(All &&... a);

private:
  std::chrono::time_point<_ClockType> start_;
  std::string delim_;
  char *time_format_;
};

template <typename U, typename C> info<U, C>::info(const std::string &&delim) {
  start_ = C::now();
  delim_ = std::move(delim);
  std::string time_format = "%5.3e";
  time_format_ = new char[time_format.size() + 1];
  std::strcpy(time_format_, time_format.c_str());
  std::cout << std::scientific << std::setprecision(3);
}

template <typename U, typename C> info<U, C>::~info() { delete[] time_format_; }

template <typename U, typename C>
void info<U, C>::set_cout_default() {
  std::cout << std::resetiosflags(std::ios_base::floatfield);
}

template <typename U, typename C>
void info<U, C>::set_delim(const std::string &&delim) {
  delim_ = std::move(delim);
}

template <typename U, typename C>
void info<U, C>::set_time_format(const std::string &&time_format) {
  delete[] time_format_;
  time_format_ = new char[time_format.size() + 1];
  std::strcpy(time_format_, time_format.c_str());
}

template <typename U, typename C> void info<U, C>::print(const char *fmt, ...) {
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);

  auto print_string_stdout = [&](const char *s) {
    fputs(s, stdout);
    fflush(stdout);
  };
  print_string_stdout(buf);
}

template <typename U, typename C>
void info<U, C>::print_time(const char *fmt, ...) {
  std::chrono::time_point<C> end = C::now();
  std::chrono::duration<double> diff = end - start_;
  char timebuf[BUFSIZ];
  sprintf(timebuf, time_format_,
          1e-3 * std::chrono::duration_cast<U>(diff).count());
  fputs(timebuf, stdout);
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);

  auto print_string_stdout = [&](const char *s) {
    fputs(s, stdout);
    fflush(stdout);
  };
  print_string_stdout(buf);
}

template <typename U, typename C>
template <class Head, class... Tail>
void info<U, C>::out(Head &&head, Tail &&... tail) {
  if (sizeof...(tail) == 0) {
    std::cout << head << std::flush;
  } else {
    std::cout << head << delim_;
  }
  outnl(std::move(tail)...);
}

template <typename U, typename C>
template <class Head, class... Tail>
void info<U, C>::outnl(Head &&head, Tail &&... tail) {
  if (sizeof...(tail) == 0) {
    std::cout << head << std::endl;
  } else {
    std::cout << head << delim_;
  }
  outnl(std::move(tail)...);
}

template <typename U, typename C>
template <class... All>
void info<U, C>::out_time(All &&... a) {
  std::chrono::time_point<C> end = C::now();
  std::chrono::duration<double> diff = end - start_;
  std::cout << 1e-3 * std::chrono::duration_cast<U>(diff).count() << delim_;
  outnl(std::move(a)...);
}
}

#endif
