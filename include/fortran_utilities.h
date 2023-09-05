#ifndef FORTRAN_UTILITIES_H
#define FORTRAN_UTILITIES_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

extern "C" {
  void f_math_invert(double* InvA, int* err, double* A, int* n);
  void f_rotate_tensor4(double* qu, double* T, double* rotated, int* active);
  void f_rotate_tensor2(double* qu, double* T, double* rotated, int* active);
  void f_math_3333to99(double* m3333, double* m99);
  void f_math_99to3333(double* m99, double* m3333);
  void f_math_mul3333xx33(double* A, double* B, double* res);
}

class FortranUtilities {
  public:

  template<typename T>
  static void invert_matrix(T& output, T& input) {
    int errmatinv = 0;
    int size_reduced = input.rows(); // assuming the matrix is square
    f_math_invert(output.data(), &errmatinv, input.data(), &size_reduced);
    if (errmatinv) {
      std::stringstream ss;
      ss << input;
      throw std::runtime_error("Matrix inversion error:\n" + ss.str());
    }
  }

  //verify that this works as expected
  template <typename T>
  static Eigen::Tensor<double, 2> rotate_tensor2(Eigen::Quaterniond& rot_input, T& rotation, bool active = false) {
    // int 0 is equivalent to fortran logical false, int -1 is equivalent to true
    int active_ = active? -1 : 0;
    Eigen::Tensor<double, 2> rotated(3, 3);
    // default eigen quaternion representation in memory is x,y,z,w -> convert to fortran representation 
    Eigen::Vector4d quaternion_f(rot_input.w(), rot_input.x(), rot_input.y(), rot_input.z());

    f_rotate_tensor2(quaternion_f.data(), rotation.data(), rotated.data(), &active_);
    return rotated;
  }

  static Eigen::Tensor<double, 4> rotate_tensor4(Eigen::Quaterniond& rot_input, Eigen::Tensor<double, 4>& rotation, bool active = false) {
    int active_ = active? -1 : 0;
    Eigen::Tensor<double, 4> rotated(3, 3, 3, 3);
    Eigen::Vector4d quaternion_f(rot_input.w(), rot_input.x(), rot_input.y(), rot_input.z());
    f_rotate_tensor4(quaternion_f.data(), rotation.data(), rotated.data(), &active_);
    return rotated;
  }
};


#endif // FORTRAN_UTILITIES_H