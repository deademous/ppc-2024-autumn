// Copyright 2024 Nesterov Alexander
#include "seq/opolin_d_simple_iteration_method/include/ops_seq.hpp"

#include <climits>
#include <cmath>
#include <random>
#include <utility>

using namespace std::chrono_literals;

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // init data
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.assign(ptr, ptr + n_);
  epsilon_ = *reinterpret_cast<double*>(taskData->inputs[2]);
  C_.resize(n_ * n_, 0.0);
  d_.resize(n_, 0.0);
  Xold.resize(n_, 0.0);
  Xnew.resize(n_, 0.0);
  max_iter_ = *reinterpret_cast<int*>(taskData->inputs[3]);
  std::vector<std::vector<double>> augmen_matrix = A_;
  for (size_t i = 0; i < n_; ++i) {
    augmen_matrix[i].push_back(b_[i]);
  }
  int rankA = rank(A_);
  int rank_augmented = rank(augmen_matrix);
  if (rankA != rank_augmented) {
    return false;
  }
  // generate C matrix and d vector
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {
        C_[i * n_ + j] = -A_[i][j] / A_[i][i];
      }
    }
    d_[i] = b_[i] / A_[i][i];
  }
  return true;
}

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  // check input and output
  if (taskData->inputs_count.empty() || taskData->inputs.size() != 4) return false;
  if (taskData->outputs_count.empty() || taskData->inputs_count[0] != taskData->outputs_count[0] ||
      taskData->outputs.empty())
    return false;

  n_ = taskData->inputs_count[0];
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.resize(n_);
  for (size_t i = 0; i < n_; i++, ptr += n_) A_[i].assign(ptr, ptr + n_);

  // check main diagonal
  for (size_t i = 0; i < n_; ++i) {
    if (std::abs(A_[i][i]) < std::numeric_limits<double>::epsilon()) {
      return false;
    }
  }
  // check method applicability
  for (size_t i = 0; i < n_; ++i) {
    double diagonal = std::abs(A_[i][i]);
    double sum_row = 0.0;
    double sum_col = 0.0;
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {
        sum_row += std::abs(A_[i][j]);
        sum_col += std::abs(A_[j][i]);
      }
    }
    if (diagonal <= sum_row && diagonal <= sum_col) {
      return false;
    }
  }
  return true;
}

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::run() {
  internal_order_test();
  // simple iteration method
  int iter = 0;
  while (iter < max_iter_) {
    for (size_t i = 0; i < n_; ++i) {
      double iter_sum = 0.0;
      for (size_t j = 0; j < n_; ++j) {
        if (i != j) {
          iter_sum += C_[i * n_ + j] * Xold[j];
        }
      }
      Xnew[i] = d_[i] + iter_sum;
    }
    double error = 0.0;
    for (size_t i = 0; i < n_; ++i) {
      error = std::max(error, std::abs(Xnew[i] - Xold[i]));
    }
    Xold = Xnew;
    if (error < epsilon_) break;
    ++iter;
  }
  return true;
}

bool opolin_d_simple_iteration_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(Xnew.begin(), Xnew.end(), out);
  return true;
}

int opolin_d_simple_iteration_method_seq::rank(std::vector<std::vector<double>> matrix) {
  size_t rowCount = matrix.size();
  if (rowCount == 0) return 0;
  size_t colCount = matrix[0].size();
  int rank = 0;
  for (size_t col = 0, row = 0; col < colCount && row < rowCount; ++col) {
    size_t maxRowIdx = row;
    double maxValue = std::abs(matrix[row][col]);
    for (size_t i = row + 1; i < rowCount; ++i) {
      double currentValue = std::abs(matrix[i][col]);
      if (currentValue > maxValue) {
        maxValue = currentValue;
        maxRowIdx = i;
      }
    }
    if (maxValue < std::numeric_limits<double>::epsilon()) continue;

    if (maxRowIdx != row) {
      for (size_t j = 0; j < colCount; ++j) {
        double temp = matrix[row][j];
        matrix[row][j] = matrix[maxRowIdx][j];
        matrix[maxRowIdx][j] = temp;
      }
    }

    double leadElement = matrix[row][col];
    for (size_t j = col; j < colCount; ++j) {
      matrix[row][j] /= leadElement;
    }

    for (size_t i = 0; i < rowCount; ++i) {
      if (i != row) {
        double factor = matrix[i][col];
        for (size_t j = col; j < colCount; ++j) {
          matrix[i][j] -= factor * matrix[row][j];
        }
      }
    }
    ++rank;
    ++row;
  }
  return rank;
}