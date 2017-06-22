#ifndef STOPT_ALGORITHM_H
#define STOPT_ALGORITHM_H

#include "adaptreg.hpp"
#include "sdca.hpp"
#include "svrg.hpp"
#include "saga.hpp"
#include "dasvrda.hpp"
#include "katyusha.hpp"
#include "spdc.hpp"

namespace stopt {

enum class algorithm : int { spdc, katyusha, dasvrda, svrg, saga, sdca };
}
#endif
