// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_cg_method/include/ops_mpi.hpp"

#include <algorithm>
#include <climits>
#include <random>
#include <utility>

using namespace std::chrono_literals;

bool opolin_d_cg_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // init data  
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.assign(ptr, ptr + n_);

  epsilon_ = *reinterpret_cast<double*>(taskData->inputs[2]);
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.empty() || taskData->inputs.size() != 3) return false;

  if (taskData->outputs_count.empty() || taskData->inputs_count[0] != taskData->outputs_count[0] ||
      taskData->outputs.empty())
    return false;

  n_ = taskData->inputs_count[0];
  if (n_ <= 0) return false;

  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(ptr, ptr + n_ * n_);

  if (!isSimmetric(A_, n_)) return false;

  if (!isPositiveDefinite(A_, n_)) return false;
  
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  x_.resize(n_);
  std::vector<double> r_k = b_; //r0 = b - Ax0
  std::vector<double> p_k = r_k;
  while (true) {
    double rsquare_prev = opolin_d_cg_method_mpi::scalarProduct(r_k, r_k);
    std::vector<double> Ap = opolin_d_cg_method_mpi::multiplyVecMat(p_k, A_);
    double alpha_k = rsquare_prev / opolin_d_cg_method_mpi::scalarProduct(p_k, Ap);
        
    // x_k+1
    for (size_t i = 0; i < n_; i++) {
      x_[i] += alpha_k * p_k[i];
    }
        
    // r_k+1
    for (size_t i = 0; i < n_; i++) {
      r_k[i] -= alpha_k * Ap[i];
    }

    double rsquare_k = opolin_d_cg_method_mpi::scalarProduct(r_k, r_k);
    // right accuracy is achieved
    if (sqrt(rsquare_k) < epsilon_) {              
      break;
    }                                             
        
    double beta_k = rsquare_k / rsquare_prev; 
    rsquare_prev = rsquare_k;
    // p_k+1
    for (size_t i = 0; i < n_; i++) {
      p_k[i] = r_k[i] + beta_k * p_k[i];
    } 
  }
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x_.begin(), x_.end(), out);
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    b_.assign(ptr, ptr + n_);

    epsilon_ = *reinterpret_cast<double*>(taskData->inputs[2]);
  }
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count.empty() || taskData->inputs.size() != 3) return false;

    if (taskData->outputs_count.empty() || taskData->inputs_count[0] != taskData->outputs_count[0] ||
        taskData->outputs.empty())
      return false;

    n_ = taskData->inputs_count[0];
    if (n_ <= 0) return false;

    auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    A_.assign(ptr, ptr + n_ * n_);

    if (!isSimmetric(A_, n_)) return false;

    if (!isPositiveDefinite(A_, n_)) return false;
  }
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  int size = world.size();

  broadcast(world, n_, 0);
  std::vector<double> local_A;
  std::vector<double> local_b;

  int chunk_size = n_ / size;
  int remaining = n_ % size;
  int local_rows = (rank == 0) ? chunk_size + remaining : chunk_size;

  if (rank == 0) {
    int offset = local_rows;
    for (int proc = 1; proc < size; ++proc) {
      int proc_rows = (proc == size - 1) ? chunk_size + remaining : chunk_size;
      std::vector<double> a_part(proc_rows * n_);
      std::vector<double> b_part(proc_rows);
      
      for (int i = 0; i < proc_rows; ++i) {
        std::copy(A_.begin() + (offset + i) * n_, A_.begin() + (offset + i + 1) * n_, a_part.begin() + i * n_);
        b_part[i] = b_[offset + i];
      }
      
      world.send(proc, 0, a_part);
      world.send(proc, 1, b_part);
      offset += proc_rows;
    }
    local_A.assign(A_.begin(), A_.begin() + local_rows * n_);
    local_b.assign(b_.begin(), b_.begin() + local_rows);
  } else {
    world.recv(0, 0, local_A);
    world.recv(0, 1, local_b);
  }
  std::vector<double> x_local(local_rows, 0.0);
  std::vector<double> r_local(local_b);
  std::vector<double> p_local(r_local);

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);
  int chunk_size = n_ / size;
  int remaining = n_ % size;
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = (i == 0) ? chunk_size + remaining : chunk_size;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
  }

  std::vector<double> global_p(n_);
  bool stop = false;
  double epsilon_sq = epsilon_ * epsilon_;

  while (!stop) {
    MPI_Allgatherv(p_local.data(), local_rows, MPI_DOUBLE, global_p.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    std::vector<double> Ap_local(local_rows, 0.0);
    for (int i = 0; i < local_rows; ++i) {
      for (int j = 0; j < n_; ++j) {
        Ap_local[i] += local_A[i * n_ + j] * global_p[j];
      }
    }

    double local_rr = 0.0;
    double local_pAp = 0.0;
    for (int i = 0; i < local_rows; ++i) {
      local_rr += r_local[i] * r_local[i];
      local_pAp += p_local[i] * Ap_local[i];
    }

    double global_rr, global_pAp;
    MPI_Allreduce(&local_rr, &global_rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double alpha = global_rr / global_pAp;

    for (int i = 0; i < local_rows; ++i) {
      x_local[i] += alpha * p_local[i];
      r_local[i] -= alpha * Ap_local[i];
    }

    double local_rr_new = 0.0;
    for (int i = 0; i < local_rows; ++i) {
      local_rr_new += r_local[i] * r_local[i];
    }

    double global_rr_new;
    MPI_Allreduce(&local_rr_new, &global_rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (global_rr_new < epsilon_sq) {
      stop = true;
      break;
    }

    double beta = global_rr_new / global_rr;
    for (int i = 0; i < local_rows; ++i) {
      p_local[i] = r_local[i] + beta * p_local[i];
    }

    global_rr = global_rr_new;
  }

  if (rank == 0) {
    x_.resize(n_);
  }
  MPI_Gatherv(x_local.data(), local_rows, MPI_DOUBLE, x_.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  return true;
}

bool opolin_d_cg_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(x_.begin(), x_.end(), out);
  }
  return true;
}

bool opolin_d_cg_method_mpi::isPositiveDefinite(const std::vector<double>& mat, size_t size) {
  std::vector<double> L(size * size, 0);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      if (j == i) {
        for (int k = 0; k < j; k++)
          sum += L[j * size + k] * L[j * size + k];
        double val = mat[j * size + j] - sum;
        if (val <= 0) return false;
        L[j * size + j] = std::sqrt(val);
      } else {
        for (int k = 0; k < j; k++)
          sum += L[i * size + k] * L[j * size + k];
        L[i * size + j] = (mat[i * size + j] - sum) / L[j * size + j];
      }
    }
  }
  return true;
}

bool opolin_d_cg_method_mpi::isSimmetric(const std::vector<double>& mat, size_t size) {
  bool simetric = true;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (j != i) {
        if (mat[i * size + j] != mat[j * size + i]) {
          simetric = false;
        }
      }
    }
  }
  return simetric;
}

double opolin_d_cg_method_mpi::scalarProduct(const std::vector<double>& a_, const std::vector<double>& b_) {
  size_t size = a_.size();
  double result = 0.0;
    for (size_t i = 0; i < size; i++) {
        result += a_[i] * b_[i];
    }
    return result;
}

std::vector<double> opolin_d_cg_method_mpi::multiplyVecMat(const std::vector<double>& vec, const std::vector<double>& mat) {
  size_t size = vec.size();
  std::vector<double> result(size, 0.0);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i] += mat[i* size + j] * vec[j];
    }
  }
  return result;
}
