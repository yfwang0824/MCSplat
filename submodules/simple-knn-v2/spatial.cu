/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

std::tuple<torch::Tensor, torch::Tensor> 
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  torch::Tensor knn_indices = torch::full({P, 3}, -1, int_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>(), knn_indices.contiguous().data<int>());

  return std::make_tuple(means, knn_indices);
}

std::tuple<torch::Tensor, torch::Tensor> 
distCUDACross(const torch::Tensor& points, const torch::Tensor& labels)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  torch::Tensor knn_indices = torch::full({P}, -1, int_opts);
  
  SimpleKNN::nearestCrossSetDistance(P, (float3*)points.contiguous().data<float>(), labels.contiguous().data<int>(), means.contiguous().data<float>(), knn_indices.contiguous().data<int>());

  return std::make_tuple(means, knn_indices);
}