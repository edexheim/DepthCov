#include "cov.h"
#include "sampler.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cross_covariance", &cross_covariance, "cross covariance");
  m.def("get_new_chol_obs_info", &get_new_chol_obs_info, "get new chol obs info");
  // m.def("greedy_entropy_sampler", &greedy_entropy_sampler, "greedy entropy sampler");
}