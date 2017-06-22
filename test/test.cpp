#include "algorithm.h"
#include "cmdline.h"

using namespace stopt;
using namespace Eigen;
using namespace std;

template <class Rerm>
void set_parameters(Rerm &r, cmdline::parser &p) {
  r.set_parameters(
      static_cast<stopt::regularization_term>(p.get<int>("regularization")),
      static_cast<stopt::loss_term>(p.get<int>("loss")),
      p.get<double>("lambda1"), p.get<double>("lambda2"),
      p.get<double>("gamma"), p.get<double>("criteria"), p.get<int>("maxitr"));
  r.set_minibatch_size(p.get<int>("minibatch"));
}

int main(int argc, char const *argv[]) {
  cmdline::parser p;
  string algo_info =
      "Algorithm:\n"
      "    0 : SPDC (Stochastic Primal Dual Coordinate)\n"
      "    1 : Katyusha\n"
      "    2 : DASVRDA (Doubly Accelerated Stochastic Variance "
      "Reduced Dual Averaging) \n"
      "    3 : SVRG (Stochastic Variance Reduced Gradient)\n"
      "    4 : SAGA (Stochastic Averaging Gradient Ameliore)\n"
      "    5 : SDCA (Stochastic Dual Coordinate Ascent)";
  string regu_info =
      "Regularization term:\n    0 : L1\n    1 : Squared\n    2 : Elastic net";
  string loss_info =
      "Loss term:\n    0 : Smoothed hinge\n    1 : Squared "
      "hinge\n    2 : Logistic\n    3 : Squared\n    4 : "
      "Smoothed insensitive";
  p.add<string>("file", 'f', "File name of training dataset (libsvm format)",
                true, "");
  p.add<int>("algorithm", 'a', algo_info, true, 0);
  p.add<int>("regularization", 'r', regu_info, false, 1);
  p.add<int>("loss", 'l', loss_info, false, 0);
  p.add<int>("minibatch", 'm', "Mini-batch size", false, 1);
  p.add<double>("criteria", 'c', "Stopping criteria (dualityt gap)", false,
                1e-9);
  p.add<double>("lambda1", 'x', "Regularization parameter of l1", false, 1e-3);
  p.add<double>("lambda2", 'y', "Regularization parameter of l2", false, 1e-3);
  p.add<double>("gamma", 'g', "Smoothness parameter of smoothed loss", false, 1.0);
  p.add<double>("epsilon", 'e', "Insensitiveness parameter of ", false, 0.1);
  p.add<bool>("adaptreg", 'p', "AdaptReg", false, false);
  p.add<int>("maxitr", 'i', "Maximum outer iteration", false, 10000);
  p.parse_check(argc, argv);

  algorithm sol = static_cast<algorithm>(p.get<int>("algorithm"));
  if (sol == algorithm::spdc) {
    spdc<double> obj(p.get<string>("file"));
    set_parameters(obj, p);
    obj.train(sampling::uniform);
  } else if (sol == algorithm::katyusha) {
    katyusha<double> obj(p.get<string>("file"));
    set_parameters(obj, p);
    obj.train();
  } else if (sol == algorithm::dasvrda) {
    dasvrda<double> obj(p.get<string>("file"));
    set_parameters(obj, p);
    obj.train();
  } else if (sol == algorithm::svrg) {
    svrg<double> obj(p.get<string>("file"));
    set_parameters(obj, p);
    if (!p.get<bool>("adaptreg")) {
      obj.train();
    } else {
      adaptreg<svrg<double>> oo;
      oo.train(obj);
    }
  } else if (sol == algorithm::saga) {
    saga<double> obj(p.get<string>("file"));
    set_parameters(obj, p);
    obj.train();
  } else if (sol == algorithm::sdca) {
    sdca<double> obj(p.get<string>("file"));
    set_parameters(obj, p);
    obj.train();
  }

  return 0;
}
