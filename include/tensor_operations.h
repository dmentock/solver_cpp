
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip> 

template <typename T, int N>
bool tensor_eq(const Eigen::Tensor<T, N>& tensor1, const Eigen::Tensor<T, N>& tensor2, double epsilon = 1e-8) {
    assert(tensor1.dimensions() == tensor2.dimensions());
    bool mismatch_found = false;
    for (Eigen::Index i = 0; i < tensor1.size(); ++i) {
        if (std::abs(tensor1.data()[i] - tensor2.data()[i]) > epsilon) {
            if (!mismatch_found) {
                std::cout << "Mismatch found between tensors:" << std::endl;
                mismatch_found = true;
            }
            Eigen::array<Eigen::Index, N> index;
            Eigen::Index idx = i;
            for (int d = 0; d < N; ++d) {
                index[d] = idx % tensor1.dimension(d);
                idx /= tensor1.dimension(d);
            }

            std::cout << "At index (";
            for (int d = 0; d < N; ++d) {
                std::cout << index[d];
                if (d < N - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::setprecision(17) << "): Tensor 1 = " << tensor1.data()[i] << ", Tensor 2 = " << tensor2.data()[i] << std::endl;
        }
    }

    if (!mismatch_found) {
        return true;
    } else {
        return false;
    }
}

template <typename T, int N>
void fill_random(Eigen::TensorMap<Eigen::Tensor<T, N>>& tensor_map) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < tensor_map.size(); ++i) {
    tensor_map.data()[i] = dis(gen);
  }
}

template <typename TensorType>
auto tensor_sum(const TensorType& tensor) -> typename TensorType::Scalar {
  using T = typename TensorType::Scalar;
  T sum = 0;
  for (int i = 0; i < tensor.size(); ++i) {
    sum += tensor.data()[i];
  }
  return sum;
}

template <typename MatrixType>
auto matrix_sum(const MatrixType& mat) -> typename MatrixType::Scalar {
  using T = typename MatrixType::Scalar;
  T sum = 0;
  for (int i = 0; i < mat.size(); ++i) {
    sum += mat.data()[i]; // for double data type
  }
  return sum;
}


// template <typename TensorType>
// bool assert_tensor_eq(const TensorType& tensor, TensorType val) {
//   using T = typename TensorType::Scalar;
//   T sum = 0;
//   for (int i = 0; i < tensor.size(); ++i) {
//     if 
//   }
//   return sum;
// }