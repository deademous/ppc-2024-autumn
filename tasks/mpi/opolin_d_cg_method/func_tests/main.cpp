// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "mpi/opolin_d_cg_method/include/ops_mpi.hpp"

void genDataCGMethod(size_t size, std::vector<double>& A, std::vector<double>& b, std::vector<double>& expectedX) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(-5.0 , 5.0);

  std::vector<double> M(size * size);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      M[i * size + j] = dist(gen);

  A.assign(size * size, 0.0);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        A[i * size + j] += M[k * size + i] * M[k * size + j];
    
  for (int i = 0; i < size; i++)
    A[i * size + i] += size;

  expectedX.resize(size);
  for (int i = 0; i < size; i++)
    expectedX[i] = dist(gen);

  b.assign(size, 0.0);
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      b[i] += A[i * size + j] * expectedX[j];
}


TEST(opolin_d_cg_method_mpi, test_small_system) {
  int size = 5;
  double epsilon = 1e-8;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  genDataCGMethod(size, A, b, x_ref);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

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
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_big_system) {
  int size = 100;
  double epsilon = 1e-8;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  genDataCGMethod(size, A, b, x_ref);

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

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
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_correct_input) {
  int size = 3;
  double epsilon = 1e-8;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  A = {29.0, 29.0, 39.0, 29.0, 53.0, 17.0, 39.0, 17.0, 90.0};
  b = {204.0, 186.0, 343.0};

  x_ref = {1.0, 2.0, 3.0};
  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

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
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-3);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_no_simetric_matrix) {
  int size = 3;
  double epsilon = 1e-8;

  boost::mpi::communicator world;
  std::vector<double> x_ref, A, b;
  A = {29.0, 0.0, 39.0, 29.0, 53.0, 17.0, 39.0, 1.0, 90.0};
  b = {0.0, 0.0, 0.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), false);
  }
}

TEST(opolin_d_cg_method_mpi, test_negative_values) {
  int size = 3;
  double epsilon = 1e-8;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  A = {244.913, -64.084, 59.893, -64.084, 84.215, -23.392, 59.893, -23.392, 31.227};
  b = {47.955, -146.484, 35.406};
  x_ref = {-0.437926, -1.924931, 0.531806};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

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
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-5);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_no_positive_define_matrix) {
  int size = 3;
  double epsilon = 1e-8;

  boost::mpi::communicator world;

  std::vector<double> A, b;
  A = {0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};
  b = {0.0, 0.0, 0.0};

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    std::vector<double> x_seq(size, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), false);
  }
}

TEST(opolin_d_cg_method_mpi, test_simple_matrix) {
  int size = 3;
  double epsilon = 1e-8;

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
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

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
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-5);
    }
  }
}

TEST(opolin_d_cg_method_mpi, test_single_element) {
  int size = 1;
  double epsilon = 1e-8;

  boost::mpi::communicator world;

  std::vector<double> x_ref, A, b;
  A = { 1.0 };
  b = { 10.0 };
  x_ref = { 10.0 };

  std::vector<double> x_out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs_count.emplace_back(x_out.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
  }

  opolin_d_cg_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMPI);

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
    taskDataSeq->inputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs_count.emplace_back(x_seq.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_seq.data()));
    opolin_d_cg_method_mpi::TestMPITaskSequential testSeq(taskDataSeq);

    ASSERT_EQ(testSeq.validation(), true);
    testSeq.pre_processing();
    testSeq.run();
    testSeq.post_processing();

    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_seq[i], x_ref[i], 1e-5);
    }
  }
}