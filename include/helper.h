#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iomanip> // Include this header for std::setw

template <typename T, int N>
Eigen::Tensor<T, N> generate_tensor(T* ndarray, Eigen::array<Eigen::Index, N> dims) {
    Eigen::TensorMap<Eigen::Tensor<const T, N, Eigen::RowMajor>> tensor_map(ndarray, dims);
    Eigen::Tensor<T, N> tensor(tensor_map.dimensions());
    tensor = tensor_map.reshape(dims).swap_layout();
    return tensor;
}

template <typename T, int N>
Eigen::Tensor<T, N> array_to_eigen_tensor(const T* data, const Eigen::array<Eigen::Index, N>& dims) {
    Eigen::Tensor<T, N> col_major_tensor(dims);
    Eigen::array<Eigen::Index, N> row_major_strides;
    Eigen::array<Eigen::Index, N> col_major_strides;
    // Compute row-major strides
    row_major_strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; --i) row_major_strides[i] = row_major_strides[i + 1] * dims[i + 1];
    // Compute column-major strides
    col_major_strides[0] = 1;
    for (int i = 1; i < N; ++i) col_major_strides[i] = col_major_strides[i - 1] * dims[i - 1];
    Eigen::array<Eigen::Index, N> index;
    for (Eigen::Index i = 0; i < col_major_tensor.size(); ++i) {
        Eigen::Index row_major_offset = 0;
        Eigen::Index col_major_offset = 0;
        for (int d = 0; d < N; ++d) {
            index[d] = (i / col_major_strides[d]) % dims[d];
            row_major_offset += index[d] * row_major_strides[d];
            col_major_offset += index[d] * col_major_strides[d];
        }
        col_major_tensor.data()[col_major_offset] = data[row_major_offset];
    }
    return col_major_tensor;
}

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
            std::cout << std::setprecision(10) << "): Tensor 1 = " << tensor1.data()[i] << ", Tensor 2 = " << tensor2.data()[i] << std::endl;
        }
    }

    if (!mismatch_found) {
        return true;
    } else {
        return false;
    }
}


//recursively print out tensor values
template <typename T, int N>
void print_tensor(const Eigen::Tensor<T, N>& tensor, std::vector<size_t>& indices, size_t level, const std::string& indent, int maxWidth) {
  if (level == N - 1) {
    std::cout << indent << "{\n";
    for (int i = 0; i < tensor.dimension(level); ++i) {
      indices[level] = i;
      std::cout << std::setprecision(10) << indent << std::setw(maxWidth) << tensor(indices);
      if (i < tensor.dimension(level) - 1) {
        std::cout << ",";
      }
    }
    std::cout << "\n" << indent << "}";
  } else {
    std::cout << indent << "{\n";
    for (int i = 0; i < tensor.dimension(level); ++i) {
      indices[level] = i;
      print_tensor(tensor, indices, level + 1, indent + " ", maxWidth);
      if (i < tensor.dimension(level) - 1) {
        std::cout << ",";
      }
    }
    std::cout << "\n" << indent << "}";
  }
}
template <typename T, int N>
int get_max_width(const Eigen::Tensor<T, N>& tensor) {
  int maxWidth = 0;
  for (int i = 0; i < tensor.size(); ++i) {
    std::stringstream ss;
    ss << tensor(i);
    int width = ss.str().length();
    maxWidth = std::max(maxWidth, width);
  }
  return maxWidth;
}
template <typename T, int N>
void print_tensor(const Eigen::Tensor<T, N>* tensor) {
  std::vector<size_t> indices(N, 0);
  int maxWidth = get_max_width(*tensor);
  print_tensor(*tensor, indices, 0, "", maxWidth); // Pass the maxWidth as an argument
}


#endif // HELPER_H