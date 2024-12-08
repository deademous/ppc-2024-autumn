// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/opolin_d_simple_iteration_method/include/ops_seq.hpp"

double getRandomDouble(double min, double max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return min + (gen() / (static_cast<double>(RAND_MAX)) * (max - min));
}

void generateTestData(int size, std::vector<double> &x, std::vector<double> &A, std::vector<double> &b) {
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

TEST(opolin_d_simple_iteration_method_seq, test_pipeline_run) {
  int size = 500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  generateTestData(size, X, A, b);
  std::vector<double> out(size, 0);
  double epsilon = 1e-5;
  int maxIters = 1000;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<opolin_d_simple_iteration_method_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(opolin_d_simple_iteration_method_seq, test_task_run) {
  int size = 500;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  generateTestData(size, X, A, b);

  std::vector<double> out(size, 0);
  double epsilon = 1e-5;
  int maxIters = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<opolin_d_simple_iteration_method_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}