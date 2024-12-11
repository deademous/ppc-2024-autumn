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

namespace opolin_d_simple_iteration_method_mpi {

int rank(std::vector<std::vector<double>> matrix);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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
  uint32_t n_;
  double epsilon_;
  int max_iters_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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
  uint32_t n_;
  double epsilon_;
  int max_iters_;
  boost::mpi::communicator world;
};

}  // namespace opolin_d_simple_iteration_method_mpi