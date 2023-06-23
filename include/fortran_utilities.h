#ifndef FORTRAN_UTILITIES_H
#define FORTRAN_UTILITIES_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

extern "C" {
  void f_math_invert(double* InvA, int* err, double* A, int* n);
  void f_rotate_tensor4(double* qu, double* T, double* rotated);
  void f_rotate_tensor2(double* qu, double* T, double* rotated);
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
      std::cerr << "Matrix inversion error: " << std::endl;
      std::cerr << input << std::endl;
      exit(1);
    }
  }

  //verify that this works as expected
  template <typename T>
  static Eigen::Tensor<double, 2> rotate_tensor2(Eigen::Quaterniond& rot_input, T& rotation) {
    Eigen::Tensor<double, 2> rotated(3, 3);
    f_rotate_tensor2(rot_input.coeffs().data(), rotation.data(), rotated.data());
    return rotated;
  }

  static Eigen::Tensor<double, 4> rotate_tensor4(Eigen::Quaterniond& rot_input, Eigen::Tensor<double, 4>& rotation) {
    Eigen::Tensor<double, 4> rotated(3, 3, 3, 3);
    f_rotate_tensor4(rot_input.coeffs().data(), rotation.data(), rotated.data());
    return rotated;
  }
};


#endif // FORTRAN_UTILITIES_H