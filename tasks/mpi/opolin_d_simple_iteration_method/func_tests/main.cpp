// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

double getRandomDouble(double min, double max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return min + (rand() / (static_cast<double>(RAND_MAX)) * (max - min));
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


TEST(opolin_d_simple_iteration_method_mpi, test_small_system) {
  uint8_t size = 5;
  double epsilon = 1e-7;
  int maxIters = 1000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  generateTestData(size, x_ref, A, b);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if(world.rank() == 0){
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_big_system) {
  uint8_t size = 100;
  double epsilon = 1e-7;
  int maxIters = 1000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  generateTestData(size, x_ref, A, b);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if(world.rank() == 0){
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_2x2_system) {
  uint8_t size = 2;
  double epsilon = 1e-7;
  int maxIters = 1000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  generateTestData(size, x_ref, A, b);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if(world.rank() == 0){
    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < size; ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}