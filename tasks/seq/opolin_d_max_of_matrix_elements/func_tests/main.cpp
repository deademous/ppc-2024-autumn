// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/opolin_d_max_of_matrix_elements/include/ops_seq.hpp"

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_100x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 100;
  const int count_cols = 100;
  int ref = INT_MAX;
  std::vector<int> out(1, INT_MIN);
  std::vector<std::vector<int>> in = opolin_d_max_of_matrix_elements_seq::getRandomMatrix(count_rows, count_cols);
  int i_c = gen() % count_cols;
  int i_r = gen() % count_rows;
  in[i_r][i_c] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ref, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_100x1) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 100;
  const int count_cols = 1;
  int ref = INT_MAX;

  std::vector<int> out(1, INT_MIN);
  std::vector<std::vector<int>> in = opolin_d_max_of_matrix_elements_seq::getRandomMatrix(count_rows, count_cols);
  int i_r = gen() % count_rows;
  in[i_r][0] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ref, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_1x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 1;
  const int count_cols = 100;
  int ref = INT_MAX;

  std::vector<int> out(1, INT_MIN);
  std::vector<std::vector<int>> in = opolin_d_max_of_matrix_elements_seq::getRandomMatrix(count_rows, count_cols);
  int i_c = gen() % count_cols;
  in[0][i_c] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ref, out[0]);
}

TEST(opolin_d_max_of_matrix_elements_seq, Test_Max_Matrix_1000x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int count_rows = 1000;
  const int count_cols = 100;
  int ref = INT_MAX;

  std::vector<int> out(1, INT_MIN);
  std::vector<std::vector<int>> in = opolin_d_max_of_matrix_elements_seq::getRandomMatrix(count_rows, count_cols);
  int i_c = gen() % count_cols;
  int i_r = gen() % count_rows;
  in[i_r][i_c] = ref;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  opolin_d_max_of_matrix_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ref, out[0]);
}