// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

#include <climits>
#include <random>
#include <utility>

using namespace std::chrono_literals;

bool opolin_d_simple_iteration_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // init data
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.assign(ptr, ptr + n_);
  epsilon_ = *reinterpret_cast<double*>(taskData->inputs[2]);
  C_.resize(n_ * n_, 0.0);
  d_.resize(n_, 0.0);
  Xold_.resize(n_, 0.0);
  Xnew_.resize(n_, 0.0);
  max_iters_ = *reinterpret_cast<int*>(taskData->inputs[3]);
  std::vector<double> augmen_matrix = A_;
  for (size_t i = 0; i < n_; ++i) {
    augmen_matrix.push_back(b_[i]);
  }
  // generate C matrix and d vector
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {
        C_[i * n_ + j] = -A_[i * n_ + j] / A_[i * n_ + i];
      }
    }
    d_[i] = b_[i] / A_[i * n_ + i];
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // check input and output
  if (taskData->inputs_count.empty() || taskData->inputs.size() != 4) return false;
  if (taskData->outputs_count.empty() || taskData->inputs_count[0] != taskData->outputs_count[0] ||
      taskData->outputs.empty())
    return false;

  n_ = taskData->inputs_count[0];
  if (n_ <= 0) return false;
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(ptr, ptr + n_ * n_);
  // check ranks
  size_t rankA = rank(A_, n_);
  if (rankA != n_) {
    return false;
  }

  // check main diagonal
  for (size_t i = 0; i < n_; ++i) {
    if (std::abs(A_[i * n_ + i]) < std::numeric_limits<double>::epsilon()) {
      return false;
    }
  }
  if (!isDiagonalDominance(A_, n_)) {
    return false;
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  // simple iteration method
  size_t iteration = 0;
  while (iteration < max_iters_) {
    for (size_t i = 0; i < n_; ++i) {
      double sum = d_[i];
      for (size_t j = 0; j < n_; ++j) {
        if (i != j) {
          sum += C_[i * n_ + j] * Xold_[j];
        }
      }
      Xnew_[i] = sum; 
    }
    double max_error = 0.0;
    for (size_t i = 0; i < n_; ++i) {
      double error = std::abs(Xnew_[i] - Xold_[i]);
      if (error > max_error) {
        max_error = error;
      }
    }
    Xold_ = Xnew_;
    if (max_error < epsilon_) { break; }
    ++iteration;
  }
  if (iteration == max_iters_) { return false; }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(Xnew_.begin(), Xnew_.end(), out);
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // init data
  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.assign(ptr, ptr + n_);
  epsilon_ = *reinterpret_cast<double*>(taskData->inputs[2]);
  C_.resize(n_ * n_, 0.0);
  d_.resize(n_, 0.0);
  Xold_.resize(n_, 0.0);
  Xnew_.resize(n_, 0.0);
  max_iters_ = *reinterpret_cast<int*>(taskData->inputs[3]);
  std::vector<double> augmen_matrix = A_;
  for (size_t i = 0; i < n_; ++i) {
    augmen_matrix.push_back(b_[i]);
  }
  // generate C matrix and d vector
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {
        C_[i * n_ + j] = -A_[i * n_ + j] / A_[i * n_ + i];
      }
    }
    d_[i] = b_[i] / A_[i * n_ + i];
  }
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // check input and output
    if (taskData->inputs_count.empty() || taskData->inputs.size() != 4) return false;
    if (taskData->outputs_count.empty() || taskData->inputs_count[0] != taskData->outputs_count[0] ||
      taskData->outputs.empty())
    return false;

    n_ = taskData->inputs_count[0];
    if (n_ <= 0) return false;
    auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    A_.assign(ptr, ptr + n_ * n_);

    // check ranks
    size_t rankA = rank(A_, n_);
    if (rankA != n_) {
      return false;
    }
    // check main diagonal
    for (size_t i = 0; i < n_; ++i) {
      if (std::abs(A_[i * n_ + i]) < std::numeric_limits<double>::epsilon()) {
        return false;
      }
    }
    if (!isDiagonalDominance(A_, n_)) {
      return false;
    }
  }
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, n_, 0);
  broadcast(world, epsilon_, 0);
  broadcast(world, max_iters_, 0);
  Xnew_.resize(n_);
  Xold_.resize(n_);

  int32_t base_rows = n_ / world.size();
  int32_t remainder = n_ % world.size();

  std::vector<int32_t> rows_per_worker(world.size());
  std::vector<int32_t> elements_per_worker(world.size());
  for (int rank = 0; rank < world.size(); ++rank) {
    rows_per_worker[rank] = base_rows + (rank < remainder ? 1 : 0);
    elements_per_worker[rank] = rows_per_worker[rank] * n_;
  }

  std::vector<double> local_C(elements_per_worker[world.rank()]);
  std::vector<double> local_d(rows_per_worker[world.rank()]);
  std::vector<double> local_X(rows_per_worker[world.rank()]);

  scatterv(world, C_, elements_per_worker, local_C.data(), 0);
  scatterv(world, d_, rows_per_worker, local_d.data(), 0);

  double global_error = 0.0;
  int iteration = 0;
  do {
    broadcast(world, Xold_, 0);

    for (int i = 0; i < rows_per_worker[world.rank()]; ++i) {
      double sum = local_d[i];
      for (size_t j = 0; j < Xold_.size(); ++j) {
        sum += local_C[i * n_ + j] * Xold_[j];
      }
      local_X[i] = sum;
    }

    gatherv(world, local_X, Xnew_.data(), rows_per_worker, 0);

    if (world.rank() == 0) {
      global_error = 0.0;
      for (size_t i = 0; i < n_; ++i) {
        double error = std::abs(Xnew_[i] - Xold_[i]);
        global_error = std::max(global_error, error);
      }
    }

    broadcast(world, global_error, 0);
    if (world.rank() == 0) Xold_ = Xnew_;
    ++iteration;
    broadcast(world, iteration, 0);
  } while (iteration < max_iters_ && global_error > epsilon_);
  return true;
}

bool opolin_d_simple_iteration_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(Xnew_.begin(), Xnew_.end(), out);
  }
  return true;
}

size_t opolin_d_simple_iteration_method_mpi::rank(std::vector<double> matrix, size_t n) {
  size_t rowCount = n;
  if (rowCount == 0) return 0;
  size_t colCount = n;
  int rank = 0;
  for (size_t col = 0, row = 0; col < colCount && row < rowCount; ++col) {
    size_t maxRowIdx = row;
    double maxValue = std::abs(matrix[row * n + col]);
    for (size_t i = row + 1; i < rowCount; ++i) {
      double currentValue = std::abs(matrix[i * n + col]);
      if (currentValue > maxValue) {
        maxValue = currentValue;
        maxRowIdx = i;
      }
    }
    if (maxValue < 1e-10) continue;

    if (maxRowIdx != row) {
      for (size_t j = 0; j < colCount; ++j) {
        double temp = matrix[row * n + j];
        matrix[row * n + j] = matrix[maxRowIdx * n + j];
        matrix[maxRowIdx * n + j] = temp;
      }
    }

    double leadElement = matrix[row * n + col];
    if (std::abs(leadElement) < 1e-10) {
      continue;
    }
    for (size_t j = col; j < colCount; ++j) {
      matrix[row * n + j] /= leadElement;
    }

    for (size_t i = 0; i < rowCount; ++i) {
      if (i != row) {
        double factor = matrix[i * n + col];
        for (size_t j = col; j < colCount; ++j) {
          matrix[i * n + j] -= factor * matrix[row * n + j];
        }
      }
    }
    ++rank;
    ++row;
    if (rank == n) {
      break;
    }
  }
  return rank;
}

bool opolin_d_simple_iteration_method_mpi::isDiagonalDominance(std::vector<double> mat, size_t dim){
  for (size_t i = 0; i < dim; i++) {
    double diagonal_value = std::abs(mat[i * dim + i]);
    double row_sum = 0.0;

    for (size_t j = 0; j < dim; j++) {
      if (j != i) {
        row_sum += std::abs(mat[i * dim + j]);
      }
    }

    if (diagonal_value <= row_sum) {
      return false;
    }
  }
  return true;
}