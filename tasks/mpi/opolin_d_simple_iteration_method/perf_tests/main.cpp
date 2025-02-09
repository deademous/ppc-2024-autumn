// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

void generateTestData(size_t size, std::vector<double> &X, std::vector<double> &A, std::vector<double> &b) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  X.resize(size);
  for (size_t i = 0; i < size; ++i) {
    X[i] = -10.0 + static_cast<double>(std::rand() % 1000) / 50.0;
  }

  A.resize(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        A[i * size + j] = -1.0 + static_cast<double>(std::rand() % 1000) / 500.0;
        sum += std::abs(A[i * size + j]);
      }
    }
    A[i * size + i] = sum + 1.0;
  }
  b.resize(size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      b[i] += A[i * size + j] * X[j];
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int size = 500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;

  if (world.rank() == 0) {
    // Create data
    generateTestData(size, X, A, b);
    double epsilon = 1e-7;
    int maxIters = 10000;
    std::vector<double> out(size, 0.0);
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(out.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  auto testMpiTaskParallel = std::make_shared<opolin_d_simple_iteration_method_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  ASSERT_EQ(testMpiTaskParallel->pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel->run(), true);
  ASSERT_EQ(testMpiTaskParallel->post_processing(), true);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int size = 500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  if (world.rank() == 0) {
    // Create data
    generateTestData(size, X, A, b);
    double epsilon = 1e-7;
    int maxIters = 10000;
    std::vector<double> out(size, 0.0);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(out.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<opolin_d_simple_iteration_method_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  ASSERT_EQ(testMpiTaskParallel->pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel->run(), true);
  ASSERT_EQ(testMpiTaskParallel->post_processing(), true);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}