// Copyright 2024 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_cg_method_mpi {

bool isPositiveDefinite(const std::vector<double>& mat, size_t size);
bool isSimmetric(const std::vector<double>& mat, size_t size);
double scalarProduct(const std::vector<double>& a_, const std::vector<double>& b_);
std::vector<double> multiplyVecMat(const std::vector<double>& vec, const std::vector<double>& mat);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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
  boost::mpi::communicator world;
};

}  // namespace opolin_d_cg_method_mpi