#include "rerm.hpp"

#include "spdc.hpp"

using namespace stopt;
using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[]) {
  info<> i{};
  string name = argv[1];
  cout << name << endl;
  spdc<double> obj(name);
  obj.set_parameters(regularization_term::elastic_net,
                     loss_term::squared_hinge, 1e-2 * obj.lambda_max_, 1e-3 * obj.lambda_max_, 1.0,
                     1e-3 * obj.lambda_max_);
  obj.num_mbs_ = 1;
  obj.train(sampling::optimality_violation);
  return 0;
}
