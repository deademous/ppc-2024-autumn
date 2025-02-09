// Copyright 2024 Nesterov Alexander
#include "seq/opolin_d_cg_method/include/ops_seq.hpp"

#include <climits>
#include <random>

using namespace std::chrono_literals;

bool opolin_d_cg_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.assign(ptr, ptr + n_);

  epsilon_ = *reinterpret_cast<double*>(taskData->inputs[2]);
  return true;
}

bool opolin_d_cg_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.empty() || taskData->inputs.size() != 3) return false;

  if (taskData->outputs_count.empty() || taskData->inputs_count[0] != taskData->outputs_count[0] ||
      taskData->outputs.empty())
    return false;

  n_ = taskData->inputs_count[0];
  if (n_ <= 0) return false;

  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(ptr, ptr + n_ * n_);

  if (!opolin_d_cg_method_seq::isSimmetric(A_, n_)) return false;

  if (!opolin_d_cg_method_seq::isPositiveDefinite(A_, n_)) return false;
  return true;
}

bool opolin_d_cg_method_seq::TestTaskSequential::run() {
  internal_order_test();
  x_.resize(n_);
  std::vector<double> r_k = b_; //r0 = b - Ax0           //
  std::vector<double> p_k = r_k;                         //
  while (true) {
    double rsquare_prev = opolin_d_cg_method_seq::scalarProduct(r_k, r_k);        //
    std::vector<double> Ap = opolin_d_cg_method_seq::multiplyVecMat(p_k, A_);
    double alpha_k = rsquare_prev / opolin_d_cg_method_seq::scalarProduct(p_k, Ap);//
        
    // x_k+1
    for (int i = 0; i < n_; i++) {
      x_[i] += alpha_k * p_k[i];
    }
        
    // r_k+1
    for (int i = 0; i < n_; i++) {
      r_k[i] -= alpha_k * Ap[i];
    }

    double rsquare_k = opolin_d_cg_method_seq::scalarProduct(r_k, r_k);     //
    // right accuracy is achieved
    if (sqrt(rsquare_k) < epsilon_) {              
      break;
    }                                             
        
    double beta_k = rsquare_k / rsquare_prev; 
    rsquare_prev = rsquare_k;                       //
    // p_k+1
    for (int i = 0; i < n_; i++) {
      p_k[i] = r_k[i] + beta_k * p_k[i];
    } 
  }
  return true;
}

bool opolin_d_cg_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x_.begin(), x_.end(), out);
  return true;
}

bool opolin_d_cg_method_seq::isPositiveDefinite(const std::vector<double>& mat, size_t size) {
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

bool opolin_d_cg_method_seq::isSimmetric(const std::vector<double>& mat, size_t size) {
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

double opolin_d_cg_method_seq::scalarProduct(const std::vector<double>& a_, const std::vector<double>& b_) {
  size_t size = a_.size();
  double result = 0.0;
    for (size_t i = 0; i < size; i++) {
        result += a_[i] * b_[i];
    }
    return result;
}

std::vector<double> opolin_d_cg_method_seq::multiplyVecMat(const std::vector<double>& vec, const std::vector<double>& mat) {
  size_t size = vec.size();
  std::vector<double> result(size, 0.0);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i] += mat[i* size + j] * vec[j];
    }
  }
  return result;
}
