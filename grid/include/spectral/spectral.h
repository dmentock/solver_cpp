#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <discretization_grid.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fftw3-mpi.h>
#include <complex>
#include <memory>
#include <cmath>
extern "C" {
    void f_math_invert(double *InvA, int *err, double *A, int *n);
    void f_rotate_tensor4(double *qu, double *T, double *rotated);
    void f_math_3333to99(double* m3333, double* m99);
    void f_math_99to3333(double* m99, double* m3333);
}
struct tNumerics {
    bool update_gamma;
    int itmin, itmax;
    double eps_div_atol, eps_div_rtol, eps_stress_atol, eps_stress_rtol;
};

class Spectral {
public:
    Spectral(DiscretizationGrid& grid_) : grid(grid_) {}
    virtual void init();
    template <int Rank>
    void set_up_fftw(ptrdiff_t& cells1_fftw, 
                     ptrdiff_t& cells1_offset, 
                     ptrdiff_t& cells2_fftw,
                     int size,
                     std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, Rank>>>& field_real,
                     std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, Rank>>>& field_fourier,
                     fftw_complex*& field_fourier_fftw,
                     int fftw_planner_flag,
                                fftw_plan &plan_forth, 
                     fftw_plan &plan_back,
                     const std::string& label);
    virtual std::array<std::complex<double>, 3> get_freq_derivative(std::array<int, 3>& k_s);
    virtual void update_coords(Eigen::Tensor<double, 5> &F, Eigen::Tensor<double, 2>& reshaped_x_n, Eigen::Tensor<double, 2>& reshaped_x_p);
    virtual void update_gamma(Eigen::Tensor<double, 4> &C);
    virtual void constitutive_response(Eigen::Tensor<double, 5> &P, 
                               Eigen::Tensor<double, 2> &P_av, 
                               Eigen::Tensor<double, 4> &C_volAvg, 
                               Eigen::Tensor<double, 4> &C_minMaxAvg,
                               Eigen::Tensor<double, 5> &F,
                               double Delta_t);
    void response_mech();
    void response_thermal();
    void response_damage();

    template <int Rank>
    void math_invert(Eigen::Matrix<double, Rank, Rank>& InvA, 
                     Eigen::Matrix<double, Rank, Rank>& A) {
        int err = 0;
        int rank_value = Rank;
        f_math_invert(InvA.data(), &err, A.data(), &rank_value);
    }
    enum derivative_ids { 
        DERIVATIVE_CONTINUOUS_ID,
        DERIVATIVE_CENTRAL_DIFF_ID,
        DERIVATIVE_FWBW_DIFF_ID
    };

    double wgt;
    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 5>>> tensorField_real;
    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>> tensorField_fourier;
    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 4>>> vectorField_real;
    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 4>>> vectorField_fourier;
    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 3>>> scalarField_real;
    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>>> scalarField_fourier;
    Eigen::Tensor<std::complex<double>, 4> xi1st;
    Eigen::Tensor<std::complex<double>, 4> xi2nd;
    Eigen::Tensor<std::complex<double>, 7> gamma_hat;
    Eigen::Tensor<double, 4> C_ref;
protected:
    derivative_ids spectral_derivative_ID;
private:
    const double TAU = 2 * M_PI;

    fftw_complex* tensorField_fourier_fftw;
    fftw_complex* vectorField_fourier_fftw;
    fftw_complex* scalarField_fourier_fftw;
    int tensor_size = 9;
    int vector_size = 3;
    int cells1_tensor;
    int cells1_offset_tensor;
    int cells0_reduced;

    fftw_plan plan_tensor_forth;
    fftw_plan plan_tensor_back;
    fftw_plan plan_vector_forth;
    fftw_plan plan_vector_back;
    fftw_plan plan_scalar_forth;
    fftw_plan plan_scalar_back;    
    
protected:
    DiscretizationGrid& grid;
};
#endif // SPECTRAL_H