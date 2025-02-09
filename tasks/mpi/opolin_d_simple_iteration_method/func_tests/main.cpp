// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <climits>
#include <random>
#include <vector>

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

TEST(opolin_d_simple_iteration_method_mpi, test_small_system) {
  boost::mpi::communicator world;
  int size = 5;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;
  generateTestData(size, x_ref, A, b);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_big_system) {
  int size = 100;
  double epsilon = 1e-8;
  int maxIters = 10000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  generateTestData(size, x_ref, A, b);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_correct_input) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  generateTestData(size, x_ref, A, b);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_no_dominance_matrix) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 1000;

  boost::mpi::communicator world;

  std::vector<double> A = {3.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0};
  std::vector<double> b = {3.0, 2.0, 2.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), false);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_negative_values) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  A = {5.0, -1.0, 2.0, -1.0, 6.0, -1.0, 2.0, -1.0, 7.0};
  b = {-9.0, -8.0, -21.0};
  x_ref = {-1.0, -2.0, -3.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-5);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-5);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_singular_matrix) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  boost::mpi::communicator world;

  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0};
  std::vector<double> b = {1.0, 2.0, 3.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), false);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_simple_matrix) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  A = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  b = {1.0, 1.0, 1.0};
  x_ref = {1.0, 1.0, 1.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-5);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-5);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_single_element) {
  int size = 1;
  double epsilon = 1e-8;
  int maxIters = 10000;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  A = {1.0};
  b = {10.0};
  x_ref = {10.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_simple_iteration_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-5);
    }
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_simple_iteration_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-5);
    }
  }
}