// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

double getRandomDouble(double min, double max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return min + (gen() / (static_cast<double>(RAND_MAX)) * (max - min));
}

void generateTestData(uint8_t size, std::vector<double>& x, std::vector<double>& A, std::vector<double>& b) {
  x.resize(size);
  for (int i = 0; i < size; ++i) {
    x[i] = getRandomDouble(-1000.0, 1000.0);
  }
  A.resize(size * size);
  for (int i = 0; i < size; ++i) {
    double rowSum = 0.0;
    for (int j = 0; j < size; ++j) {
      if (i != j) {
        A[i * size + j] = getRandomDouble(-500.0, 500.0);
        rowSum += std::abs(A[i * size + j]);
      }
    }
    A[i * size + i] = rowSum + getRandomDouble(1.0, 10.0);
  }
  b.resize(size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      b[i] += A[i * size + j] * x[j];
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  uint8_t size = 500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;

  if (world.rank() == 0) {
    // Create data
    generateTestData(size ,X ,A ,b);
    double epsilon = 1e-5;
    int maxIters = 1000;
    std::vector<double> out(size, 0.0);
    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(out.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  auto testMpiTaskParallel = std::make_shared<opolin_d_simple_iteration_method_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

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

   uint8_t size = 500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  if (world.rank() == 0) {
    // Create data
    generateTestData(size ,X ,A ,b);
    double epsilon = 1e-5;
    int maxIters = 1000;
    std::vector<double> out(size, 0.0);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(out.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<opolin_d_simple_iteration_method_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

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