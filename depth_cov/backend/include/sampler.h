#pragma once

#include <torch/extension.h>

torch::Tensor greedy_entropy_sampler(torch::Tensor x1, torch::Tensor E1, torch::Tensor x2, torch::Tensor E2, float scale);