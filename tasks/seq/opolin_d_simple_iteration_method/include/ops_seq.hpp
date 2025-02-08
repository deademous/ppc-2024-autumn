// Copyright 2024 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_simple_iteration_method_seq {

size_t rank(std::vector<double> matrix, size_t n);
bool isDiagonalDominance(std::vector<double> mat, size_t dim);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> C_;
  std::vector<double> b_;
  std::vector<double> d_;
  std::vector<double> Xold_;
  std::vector<double> Xnew_;
  double epsilon_;
  uint32_t n_;
  int max_iter_;
};

}  // namespace opolin_d_simple_iteration_method_seq