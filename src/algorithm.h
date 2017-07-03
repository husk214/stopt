#ifndef STOPT_ALGORITHM_H
#define STOPT_ALGORITHM_H

#include "acc_svrg.hpp"
#include "adaptreg.hpp"
#include "dasvrda.hpp"
#include "katyusha.hpp"
#include "saga.hpp"
#include "sdca.hpp"
#include "spdc.hpp"
#include "svrg.hpp"

namespace stopt {

enum class algorithm : int {
  spdc,
  katyusha,
  dasvrda,
  svrg,
  saga,
  sdca,
  acc_svrg
};
}
#endif
