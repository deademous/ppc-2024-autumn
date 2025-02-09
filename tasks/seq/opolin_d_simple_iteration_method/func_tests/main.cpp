// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/opolin_d_simple_iteration_method/include/ops_seq.hpp"

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

TEST(opolin_d_simple_iteration_method_seq, test_small_system) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int size = 3;
  std::vector<double> expectedX, A, b;
  generateTestData(size, expectedX, A, b);

  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 10000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_big_system) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int size = 100;
  std::vector<double> expectedX, A, b;
  generateTestData(size, expectedX, A, b);

  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_negative_values) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int size = 3;
  std::vector<double> expectedX, A, b;

  A = {5.0, -1.0, 2.0, -1.0, 6.0, -1.0, 2.0, -1.0, 7.0};
  b = {-9.0, -8.0, -21.0};
  expectedX = {-1.0, -2.0, -3.0};

  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(expectedX[i], out[i], 1e-3);
  }
}

TEST(opolin_d_simple_iteration_method_seq, test_single_element) {
  std::vector<double> A = {4.0};
  std::vector<double> b = {8.0};
  std::vector<double> expectedX = {2.0};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_NEAR(expectedX[0], out[0], 1e-3);
}

TEST(opolin_d_simple_iteration_method_seq, test_no_dominance_matrix) {
  std::vector<double> A = {3.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0};
  std::vector<double> b = {3.0, 2.0, 2.0};
  std::vector<double> out(3, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(opolin_d_simple_iteration_method_seq, test_singular_matrix) {
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0};
  std::vector<double> b = {1.0, 2.0, 3.0};
  std::vector<double> out(3, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(opolin_d_simple_iteration_method_seq, test_random_generated_data) {
  int size = 5;
  std::vector<double> expectedX, A, b;
  generateTestData(size, expectedX, A, b);

  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_NEAR(expectedX[0], out[0], 1e-4);
}

TEST(opolin_d_simple_iteration_method_seq, test_correct_input) {
  int size = 3;
  std::vector<double> expectedX, A, b;

  A = {4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0};
  b = {7.0, 7.0, 8.0};
  expectedX = {1.0, 1.0, 1.0};

  std::vector<double> out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_NEAR(expectedX[0], out[0], 1e-4);
}

TEST(opolin_d_simple_iteration_method_seq, test_simple_matrixs) {
  int size = 3;
  std::vector<double> expectedX, A, b;

  A = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  b = {1.0, 1.0, 1.0};
  expectedX = {1.0, 1.0, 1.0};

  std::vector<double> out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  double epsilon = 1e-9;
  int maxIters = 1000;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  opolin_d_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_NEAR(expectedX[0], out[0], 1e-4);
}