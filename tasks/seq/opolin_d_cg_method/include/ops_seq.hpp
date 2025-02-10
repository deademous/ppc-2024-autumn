// Copyright 2023 Nesterov Alexander
#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_cg_method_seq {
bool isPositiveDefinite(const std::vector<double>& mat, size_t size);
bool isSimmetric(const std::vector<double>& mat, size_t size);
double scalarProduct(const std::vector<double>& a_, const std::vector<double>& b_);
std::vector<double> multiplyVecMat(const std::vector<double>& vec, const std::vector<double>& mat);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  size_t n_;
  double epsilon_;  
};

}  // namespace opolin_d_cg_method_seq