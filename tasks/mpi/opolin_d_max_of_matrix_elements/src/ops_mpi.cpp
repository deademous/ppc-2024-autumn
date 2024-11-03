// Copyright 2023 Nesterov Alexander
#include "mpi/opolin_d_max_of_matrix_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> opolin_d_max_of_matrix_elements_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

std::vector<std::vector<int>> opolin_d_max_of_matrix_elements_mpi::getRandomMatrix(int rows, int cols) {
  std::vector<std::vector<int>> matr(rows);
  for (int i = 0; i < rows; i++) {
    matr[i] = opolin_d_max_of_matrix_elements_mpi::getRandomVector(cols);
  }
  return matr;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<int> rows_maxs(input_.size());
  for (unsigned int i = 0; i < input_.size(); i++) {
    int max_in_row = input_[i][0];
    for (unsigned int j = 1; j < input_[i].size(); j++) {
      if (input_[i][j] > max_in_row) max_in_row = input_[i][j];
    }
    rows_maxs[i] = max_in_row;
  }
  int max_in_local = rows_maxs[0];
  for (unsigned int i = 1; i < rows_maxs.size(); i++) {
    if (rows_maxs[i] > max_in_local) {
      max_in_local = rows_maxs[i];
    }
  }
  res_ = max_in_local;
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];
    input_ = std::vector<int>(rows * cols);

    for (unsigned int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < cols; j++) {
        input_[i * cols + j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int max_in_vec = local_input_[0];
  for (unsigned int i = 1; i < local_input_.size(); i++) {
    if (local_input_[i] > max_in_vec) max_in_vec = local_input_[i];
  }
  reduce(world, max_in_vec, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}