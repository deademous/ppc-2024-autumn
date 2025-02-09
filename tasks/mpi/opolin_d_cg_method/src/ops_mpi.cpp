// Copyright 2024 Nesterov Alexander
#include "mpi/opolin_d_cg_method/include/ops_mpi.hpp"

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
  std::vector<double> r_k = b_; //r0 = b - Ax0           //
  std::vector<double> p_k = r_k;                         //
  while (true) {
    double rsquare_prev = opolin_d_cg_method_mpi::scalarProduct(r_k, r_k);        //
    std::vector<double> Ap = opolin_d_cg_method_mpi::multiplyVecMat(p_k, A_);
    double alpha_k = rsquare_prev / opolin_d_cg_method_mpi::scalarProduct(p_k, Ap);//
        
    // x_k+1
    for (int i = 0; i < n_; i++) {
      x_[i] += alpha_k * p_k[i];
    }
        
    // r_k+1
    for (int i = 0; i < n_; i++) {
      r_k[i] -= alpha_k * Ap[i];
    }

    double rsquare_k = opolin_d_cg_method_mpi::scalarProduct(r_k, r_k);     //
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
  int wr = world.rank(), ws = world.size();
  boost::mpi::broadcast(world, n_, 0);
  boost::mpi::broadcast(world, epsilon_, 0);
  boost::mpi::broadcast(world, b_, 0);
  size_t chunk = n_ / ws, rem = n_ % ws;
  size_t start = (wr < rem) ? wr*(chunk+1) : rem*(chunk+1) + (wr-rem)*chunk;
  size_t end = (wr < rem) ? start + (chunk+1) : start + chunk, local_rows = end - start;
  
  std::vector<double> local_A;
  if(wr==0) {
    std::vector<std::vector<double>> scat(ws);
    for (int p = 0; p < ws; p++) {
      size_t s = (p < rem) ? p*(chunk+1) : rem*(chunk+1) + (p-rem)*chunk;
      size_t e = (p < rem) ? s + (chunk+1) : s + chunk;
      scat[p].resize((e-s)*n_);
      for (size_t i = s; i < e; i++)
        std::copy(A_.begin() + i*n_, A_.begin() + (i+1)*n_, scat[p].begin() + (i-s)*n_);
    }
    boost::mpi::scatter(world, scat, local_A, 0);
  } else {
    boost::mpi::scatter(world, local_A, 0);
  }

  std::vector<double> x(n_, 0.0), r = b_, p = r;
  double rsq_prev = 0, rsq_new = 0, alpha = 0, beta = 0, dot_p_Ap = 0;
  bool end_flag = false;

  while (true) {
    boost::mpi::broadcast(world, p, 0);
    std::vector<double> local_Ap(local_rows, 0.0);
    for (size_t i = 0; i < local_rows; i++) {
      double sum = 0;
      for (size_t j = 0; j < n_; j++)
        sum += local_A[i*n_+j] * p[j];
      local_Ap[i] = sum;
    }
    std::vector<std::vector<double>> gAp;
    boost::mpi::gather(world, local_Ap, gAp, 0);
    std::vector<double> Ap;
    if(wr==0) {
      Ap.resize(n_);
      for (int p_ = 0; p_ < ws; p_++) {
        size_t s = (p_ < rem) ? p_*(chunk+1) : rem*(chunk+1) + (p_-rem)*chunk;
        std::copy(gAp[p_].begin(), gAp[p_].end(), Ap.begin()+s);
      }
    }
    boost::mpi::broadcast(world, Ap, 0);
    if(wr==0) {
      rsq_prev = 0; for (double v : r) rsq_prev += v*v;
      dot_p_Ap = 0; for (size_t i = 0; i < p.size(); i++) dot_p_Ap += p[i] * Ap[i];
      alpha = rsq_prev / dot_p_Ap;
    }
    boost::mpi::broadcast(world, alpha, 0);
    
    // Определяем локальный диапазон для обновления векторов
    std::vector<double> x_local, r_local, p_local, Ap_local;
    if(wr==0) {
      std::vector<std::vector<double>> sx(ws), sr(ws), sp(ws), sAp(ws);
      for (int p_ = 0; p_ < ws; p_++) {
        size_t s = (p_ < rem) ? p_*(chunk+1) : rem*(chunk+1) + (p_-rem)*chunk;
        size_t e = (p_ < rem) ? s + (chunk+1) : s + chunk;
        sx[p_].assign(x.begin() + s, x.begin() + e);
        sr[p_].assign(r.begin() + s, r.begin() + e);
        sp[p_].assign(p.begin() + s, p.begin() + e);
        sAp[p_].assign(Ap.begin() + s, Ap.begin() + e);
      }
      boost::mpi::scatter(world, sx, x_local, 0);
      boost::mpi::scatter(world, sr, r_local, 0);
      boost::mpi::scatter(world, sp, p_local, 0);
      boost::mpi::scatter(world, sAp, Ap_local, 0);
    } else {
      boost::mpi::scatter(world, x_local, 0);
      boost::mpi::scatter(world, r_local, 0);
      boost::mpi::scatter(world, p_local, 0);
      boost::mpi::scatter(world, Ap_local, 0);
    }
    for (size_t i = 0; i < x_local.size(); i++) {
      x_local[i] += alpha * p_local[i];
      r_local[i] -= alpha * Ap_local[i];
    }
    double local_rsq = 0; for (double v : r_local) local_rsq += v*v;
    std::vector<std::vector<double>> gx, gr;
    boost::mpi::gather(world, x_local, gx, 0);
    boost::mpi::gather(world, r_local, gr, 0);
    if(wr==0) {
      for (int p_ = 0; p_ < ws; p_++) {
        size_t s = (p_ < rem) ? p_*(chunk+1) : rem*(chunk+1) + (p_-rem)*chunk;
        std::copy(gx[p_].begin(), gx[p_].end(), x.begin()+s);
        std::copy(gr[p_].begin(), gr[p_].end(), r.begin()+s);
      }
    }
    boost::mpi::all_reduce(world, local_rsq, rsq_new, std::plus<double>());
    if(wr==0) end_flag = (std::sqrt(rsq_new) < epsilon_);
    boost::mpi::broadcast(world, end_flag, 0);
    if(end_flag) { boost::mpi::broadcast(world, x, 0); boost::mpi::broadcast(world, r, 0); break; }
    if(wr==0) {
      beta = rsq_new / rsq_prev;
      for (size_t i = 0; i < p.size(); i++) p[i] = r[i] + beta * p[i];
    }
    boost::mpi::broadcast(world, p, 0);
  }
  if(wr==0) x_ = x;
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
