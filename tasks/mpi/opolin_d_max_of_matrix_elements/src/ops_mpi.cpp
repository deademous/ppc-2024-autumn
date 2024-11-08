// Copyright 2023 Nesterov Alexander
#include "mpi/opolin_d_max_of_matrix_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> getRandomVectorForGetMaxInMatrix(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrixForGetMaxInMatrix(int rows, int cols, int min, int max) {
  std::vector<std::vector<int>> matr(rows);
  for (int i = 0; i < rows; i++) {
    matr[i] = getRandomVectorForGetMaxInMatrix(cols, min, max);
  }
  return matr;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  unsigned int rows = taskData->inputs_count[0];
  unsigned int cols = taskData->inputs_count[1];
  input_ = std::vector<std::vector<int>>(rows, std::vector<int>(cols));
  for (unsigned int i = 0; i < rows; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < cols; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  // Init value for output
  res = std::numeric_limits<int32_t>::min();
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check non empty input data
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    for (size_t j = 0; j < input_[i].size(); j++) {
      if (input_[i][j] > res) {
        res = input_[i][j];
      }
    }
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
      return false;
    }
    // Init vectors
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];
    unsigned int total_elements = rows * cols;
    input_ = std::vector<int>(total_elements);
    // Init input vector
    for (unsigned int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < cols; j++) {
        input_[i * cols + j] = tmp_ptr[j];
      }
    }
  }
  // Init value for output
  res = std::numeric_limits<int32_t>::min();
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check non empty input data
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int total_elements = taskData->inputs_count[0] * taskData->inputs_count[1];
  unsigned int delta = total_elements / world.size();
  unsigned int remaining = total_elements % world.size();
  if (delta == 0) {
    delta = (static_cast<unsigned int>(world.rank()) < total_elements) ? 1 : 0;
  } else if (static_cast<unsigned int>(world.rank()) < remaining) {
    delta++;
  }
  unsigned int start_index = world.rank() * delta;
  unsigned int end_index = std::min((world.rank() + 1) * delta, total_elements);
  if (start_index < total_elements) {
        local_input_.resize(end_index - start_index);
        std::copy(input_.begin() + start_index, input_.begin() + end_index, local_input_.begin());
    } else {
        local_input_.clear();
    }
  int local_max =
    (local_input_.empty()) ? std::numeric_limits<int32_t>::min() : *std::max_element(local_input_.begin(), local_input_.end());
  boost::mpi::reduce(world, local_max, res, boost::mpi::maximum<int>(), 0);
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}