#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iomanip> // Include this header for std::setw

//recursively print out tensor values
template <typename T, int N>
void print(const Eigen::Tensor<T, N>& tensor, std::vector<size_t>& indices, size_t level, const std::string& indent, int maxWidth, bool compressed = false) {
  if (level == N - 1) {
    std::cout << indent << "{\n";
    for (int i = 0; i < tensor.dimension(level); ++i) {
      indices[level] = i;
      std::cout << std::setprecision(17) << indent << std::setw(maxWidth) << tensor(indices);
      if (i < tensor.dimension(level) - 1) {
        std::cout << ",";
      }
    }
    std::cout << "\n" << indent << "}";
  } else {
    std::cout << indent << "{\n";
    for (int i = 0; i < tensor.dimension(level); ++i) {
      indices[level] = i;
      print(tensor, indices, level + 1, indent + " ", maxWidth);
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
void print(const std::string& label, const Eigen::Tensor<T, N>& tensor) {
  std::cout << std::endl << "printing tensor " << label << ", dims (";
  for (int i = 0; i < tensor.dimensions().size(); ++i) {
    std::cout << tensor.dimensions()[i];
    if (i < tensor.dimensions().size() - 1) {
        std::cout << ", ";
    }
  }
  std::cout << ")" << std::endl;
  std::vector<size_t> indices(N, 0);
  int maxWidth = get_max_width(tensor);
  print(tensor, indices, 0, "", maxWidth); // Pass the maxWidth as an argument
  std::cout << std::endl;
}

template <typename T, int N>
void print_map(const std::string& label, const Eigen::TensorMap<Eigen::Tensor<T, N>>& tensor_map) {
  Eigen::Tensor<T, N> tensor = tensor_map;
  print(label, &tensor);
}

template <typename TensorType>
void cat_print_recursive(const TensorType& tensor, const std::vector<int>& indices, int current_index) {
    if (current_index == indices.size()) {
        std::cout << std::setprecision(17) <<  tensor.coeff(indices) << std::endl;
        return;
    }

    for (int i = 0; i < tensor.dimension(current_index); ++i) {
        std::vector<int> new_indices = indices;
        new_indices[current_index] = i;
        cat_print_recursive(tensor, new_indices, current_index + 1);
    }
}

template <typename TensorType>
void cat_print(const std::string& label, const TensorType& tensor) {
    std::cout << std::endl << "cat-printing tensor " << label << ", dims (";
    for (int i = 0; i < tensor.dimensions().size(); ++i) {
        std::cout << tensor.dimensions()[i];
        if (i < tensor.dimensions().size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
    std::vector<int> indices(tensor.dimensions().size(), 0);
    cat_print_recursive(tensor, indices, 0);
}


template <typename T, int Rank>
void print_f(const std::string& label, const Eigen::Tensor<T, Rank>& tensor) {
    std::cout << std::endl << "fortranstyle-printing tensor " << label << ", dims (";
    for (int i = 0; i < tensor.dimensions().size(); ++i) {
        std::cout << tensor.dimensions()[i];
        if (i < tensor.dimensions().size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
    std::array<int, Rank> index{};
    int total_elements = tensor.size();
    for (int linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
        int remainder = linear_idx;
        for (int r = 0; r < Rank; ++r) {
            index[r] = remainder % tensor.dimension(r);
            remainder /= tensor.dimension(r);
        }
        std::cout << "Element ("; 
        for (int r = 0; r < Rank; ++r) {
            std::cout << index[r] + 1;  // Use 1-based indices like Fortran
            if (r < Rank - 1) std::cout << ", ";
        }
        std::cout << std::setprecision(17) << "): " << tensor.coeff(linear_idx) << std::endl;
    }
}

template <typename T, int Rank>
void print_fp(const std::string& label, const Eigen::Tensor<T, Rank>& tensor) {
  std::cout << label << "  ";
  for (int i = 0; i < tensor.dimensions().size(); ++i) {
    std::cout << tensor.dimensions()[i] << "        ";
  }
  std::cout << "vals:    ";
  for (int linear_idx = 0; linear_idx < tensor.size(); ++linear_idx) {
    std::cout << std::setprecision(17) << tensor.coeff(linear_idx) << "        ";
  }
  std::cout << std::endl;
} 


template <typename T, int N>
void print_f_map(const std::string& label, const Eigen::TensorMap<Eigen::Tensor<T, N>>& tensor_map) {
  Eigen::Tensor<T, N> tensor = tensor_map;
  print_f(label, tensor);
}

template <typename T, int N>
void print_f_mat(const std::string& label, Eigen::Matrix<T, N, N>& mat){
    Eigen::TensorMap<Eigen::Tensor<T, 2>> tensor_map(mat.data(), N, N);
    print_f_map(label, tensor_map);
}

// template <typename T>
// void print_f_matx(const std::string& label, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
//     Eigen::Tensor<T, 2>::Dimensions dims(mat.rows(), mat.cols());
//     Eigen::TensorMap<const Eigen::Tensor<T, 2>> tensor_map(mat.data(), dims);
//     print_f_map(label, tensor_map);
// }
template <typename T, int N>
Eigen::Tensor<T, 2> mat_to_tensor(Eigen::Matrix<T, N, N> mat){
    Eigen::TensorMap<Eigen::Tensor<T, 2>> tensor_map(mat.data(), N, N);
    Eigen::Tensor<T, 2> tensor = tensor_map;
    return tensor;
}

#endif // HELPER_H