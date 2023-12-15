#ifndef UTILITIES_TENSOR_H
#define UTILITIES_TENSOR_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iomanip>

using namespace std;
using namespace Eigen;

template <typename T, int rows, int cols>
Matrix<T, rows, cols> as_matrix(const T& scalar) {
    Matrix<T, rows, cols> mat(rows, cols);
    mat.fill(static_cast<double>(scalar));
    return mat;
}

template <typename T, int rows, int cols>
Matrix<T, rows, cols> as_matrix(const Tensor<T, 2>& tensor) {
    Matrix<T, rows, cols> mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = tensor(i, j);
    return mat;
}

template <typename T, int rows, int cols>
Eigen::Matrix<T, rows, cols> merge(
    const Eigen::Matrix<T, rows, cols>& A,
    const Eigen::Matrix<T, rows, cols>& B,
    const Eigen::Matrix<bool, rows, cols>& mask)
{
    Eigen::Matrix<T, rows, cols> result;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (mask(i, j) == true) {
              result(i, j) = A(i, j);
            } else {
              result(i, j) = B(i, j);
            }
        }
    }
    return result;
}

template <typename T, int rows, int cols>
Eigen::Matrix<T, rows, cols> merge(
    const Eigen::Matrix<T, rows, cols>& A,
    T scalar,
    const Eigen::Matrix<bool, rows, cols>& mask)
{
    Eigen::Matrix<T, rows, cols> B = Eigen::Matrix<T, rows, cols>::Constant(rows, cols, scalar);
    return merge(A, B, mask);
}

template <typename T, int rows, int cols>
Eigen::Matrix<T, rows, cols> merge(
    T scalar,
    const Eigen::Matrix<T, rows, cols>& B,
    const Eigen::Matrix<bool, rows, cols>& mask)
{
    Eigen::Matrix<T, rows, cols> A = Eigen::Matrix<T, rows, cols>::Constant(rows, cols, scalar);
    return merge(A, B, mask);
}

template <typename T, int rows, int cols>
Eigen::Matrix<T, rows, cols> merge(
    const Eigen::Matrix<T, rows, cols>& A,
    T scalar,
    bool mask_values)
{
    Eigen::Matrix<T, rows, cols> B = Eigen::Matrix<T, rows, cols>::Constant(rows, cols, scalar);
    Eigen::Matrix<bool, rows, cols> mask = Eigen::Matrix<bool, rows, cols>::Constant(rows, cols, mask_values);
    return merge(A, B, mask);
}

template <typename T, int rows, int cols>
Eigen::Matrix<T, rows, cols> merge(
    T scalar,
    const Eigen::Matrix<T, rows, cols>& B,
    bool mask_values)
{
    Eigen::Matrix<T, rows, cols> A = Eigen::Matrix<T, rows, cols>::Constant(rows, cols, scalar);
    Eigen::Matrix<bool, rows, cols> mask = Eigen::Matrix<bool, rows, cols>::Constant(rows, cols, mask_values);
    return merge(A, B, mask);
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

// fortran style print
template <typename T, int Rank>
void print_f(const std::string& label, const Tensor<T, Rank>& tensor) {
    std::cout << " " << label << "   ";
    int total_elements = tensor.size();
    for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        T value = tensor.coeff(linear_idx);
        
        if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
            // Print complex number
            std::cout << std::setw(18) << std::fixed << std::setprecision(2) << value;
        } else {
            // Print real number
            int int_part = static_cast<int>(value);
            int num_digits = 1;
            while (int_part /= 10) ++num_digits;
            int precision = 18 - num_digits - 1;  // subtracting 1 for the dot

            if (linear_idx > 0) {
                if (value < 0) {
                    std::cout  << "       ";
                } else {
                    std::cout  << "        ";
                }
            }
            std::cout << std::setw(18) << std::fixed << std::setprecision(precision) << value;
        }
    }
    std::cout << "     " << std::endl;
}

// Wrapper for Matrix
template <typename T, int rows, int cols>
void print_f(const std::string& label, const Matrix<T, rows, cols>& matrix) {
    Tensor<T, 2> tensor(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j)
            tensor(i, j) = matrix(i, j);
    print_f(label, tensor);
}

// Wrapper for TensorMap (assuming Rank 2 for simplicity, can be templated further)
template <typename T, int dims>
void print_f(const std::string& label, const TensorMap<Tensor<T, dims>>& tensorMap) {
    Tensor<T, dims> tensor = tensorMap;
    print_f(label, tensor);
}

// Wrapper for Map (for Matrix)
template <typename T, int rows, int cols>
void print_f(const std::string& label, const Map<Matrix<T, rows, cols>>& map) {
    Matrix<T, rows, cols> matrix = map;
    print_f(label, matrix);
}

template<typename T, int Rank>
void print_f_raw(const std::string& label, const Tensor<T, Rank>& tensor) {
    std::cout << " " << label << "  ";
    for (int i = 0; i < tensor.size(); ++i) {
        // std::cout << std::fixed << std::setprecision(16) << tensor(i) << "        ";
        std::cout << tensor(i) << "        ";
    }
    std::cout << std::endl;
}

// converts Matrix to Tensor
template <typename T, int N>
Tensor<T, 2> mat_to_tensor(Matrix<T, N, N> mat){
    TensorMap<Tensor<T, 2>> tensor_map(mat.data(), N, N);
    Tensor<T, 2> tensor = tensor_map;
    return tensor;
}

template <typename Type>
Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> tensor_to_mat(Eigen::Tensor<Type, 2>& tensor){
  return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
}

// element-wise equality of two tensors
template <typename T1, typename T2>
bool tensor_eq(const T1& tensor1, const T2& tensor2, double epsilon = 1e-8) {
    assert(tensor1.dimensions() == tensor2.dimensions());
    bool mismatch_found = false;
    for (Eigen::Index i = 0; i < tensor1.size(); ++i) {
        if (std::abs(tensor1.data()[i] - tensor2.data()[i]) > epsilon) {
            if (!mismatch_found) {
                std::cout << "Mismatch found between tensors:" << std::endl;
                mismatch_found = true;
            }
            Eigen::array<Eigen::Index, T1::NumDimensions> index;
            Eigen::Index idx = i;
            for (int d = 0; d < T1::NumDimensions; ++d) {
                index[d] = idx % tensor1.dimension(d);
                idx /= tensor1.dimension(d);
            }

            std::cout << "At index (";
            for (int d = 0; d < T1::NumDimensions; ++d) {
                std::cout << index[d];
                if (d < T1::NumDimensions - 1) {
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

#endif // UTILITIES_TENSOR_H
