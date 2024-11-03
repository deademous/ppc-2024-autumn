// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/opolin_d_max_of_matrix_elements/include/ops_seq.hpp"

TEST(opolin_d_max_of_matrix_elements_seq, test_pipeline_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  int ref = INT_MAX;

  std::random_device dev;
  std::mt19937 gen(dev());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 2000;
  int count_cols = 2000;
  global_matrix = opolin_d_max_of_matrix_elements_seq::getRandomMatrix(count_rows, count_cols);
  int i_c = gen() % count_cols;
  int i_r = gen() % count_rows;
  global_matrix[i_r][i_c] = ref;
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  auto testTaskSequential = std::make_shared<opolin_d_max_of_matrix_elements_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(ref, global_max[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, test_task_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  int ref = INT_MAX;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 2000;
  int count_cols = 2000;

  global_matrix = opolin_d_max_of_matrix_elements_seq::getRandomMatrix(count_rows, count_cols);
  int i_c = gen() % count_cols;
  int i_r = gen() % count_rows;
  global_matrix[i_r][i_c] = ref;
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  auto testTaskSequential = std::make_shared<opolin_d_max_of_matrix_elements_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(ref, global_max[0]);
}