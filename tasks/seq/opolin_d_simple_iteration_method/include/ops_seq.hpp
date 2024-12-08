// Copyright 2024 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_simple_iteration_method_seq {

int rank(std::vector<std::vector<double>> matrix);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<double> C_;
  std::vector<double> b_;
  std::vector<double> d_;
  std::vector<double> Xold;
  std::vector<double> Xnew;
  double epsilon_;
  uint32_t n_;
  int max_iter_;
};

}  // namespace opolin_d_simple_iteration_method_seq