#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "spectral/spectral.h"

#include "simple_grid_setup.hpp"
#include "init_environments.hpp"

#include <unsupported/Eigen/CXX11/Tensor>
#include <fftw3-mpi.h>

#include <test/complex_concatenator.h>
#include <tensor_operations.h>
#include <helper.h>

TEST(TestFFT, TestInitForwardBackward) {
  fftw_mpi_init();
  std::array<int, 3> cells = {2, 1, 1};
  std::vector<int> extra_dims = {3, 3};
  FFT<5> fft_obj;
  ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
  fft_obj.init_fft(cells, 1, extra_dims, 0, &cells1_fftw, &cells1_offset, &cells2_fftw);
  Eigen::Tensor<double, 5> field_real_test(3,3,2,1,1);
  field_real_test.setRandom();
  fft_obj.set_field_real(field_real_test);
  fft_obj.forward();

  Eigen::Tensor<std::complex<double>, 5> tensorField_fourier = fft_obj.get_field_fourier();
  for (int i = 0; i < tensorField_fourier.size(); ++i) {
    if (std::abs(tensorField_fourier.data()[i].real() - 1.0) < 1e-12) {
      std::cerr << "Mismatch avg tensorField FFT <-> real" << std::endl;
      exit(-1);
    }
  }

  double wgt = 0.5;
  fft_obj.backward(wgt);
  Eigen::Tensor<double, 5> tensorField_real_after = fft_obj.get_field_real().slice(
    Eigen::array<Eigen::Index, 5>({0,0,0,0,0}), 
    Eigen::array<Eigen::Index, 5>({3,3,2,1,1}));
  EXPECT_TRUE(tensor_eq(field_real_test, tensorField_real_after));
}

TEST_F(SimpleGridSetup, SpectralTestInit) {
  gridSetup_init_grid(std::array<int, 3>{2,1,1});
  gridSetup_init_discretization();
  Spectral spectral(config, *mock_grid);

  Eigen::Tensor<std::complex<double>, 4> expected_xi1st(3, 2, 1, 1);
  expected_xi1st.setValues({
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}}
  });

  Eigen::Tensor<std::complex<double>, 4> expected_xi2nd;
  expected_xi2nd.resize(3, 2, 1, 1);
  expected_xi2nd.setValues({
   {{{ c( 0               ,  0                ) }},
    {{ c( 0               ,  314159.2653589793) }}},
   {{{ c( 0               ,  0                ) }},
    {{ c( 0               ,  0                ) }}},
   {{{ c( 0               ,  0                ) }},
    {{ c( 0               ,  0                ) }}}
  });

  spectral.init();
  EXPECT_TRUE(tensor_eq(spectral.xi1st, expected_xi1st));
  EXPECT_TRUE(tensor_eq(spectral.xi2nd, expected_xi2nd));
  // TODO: mock calls to set_up_fftw template function
}

TEST_F(SimpleGridSetup, SpectralTestHomogenizationFetchTensors) {
  gridSetup_init_grid(std::array<int, 3>{2,1,1});
  gridSetup_init_discretization();
  f_homogenization_init();
  Spectral spectral(config, *mock_grid);
  spectral.homogenization_fetch_tensor_pointers();
  f_deallocate_resources();
}

TEST_F(SimpleGridSetup, SpectralTestConstitutiveResponse) {
  class PartialMockSpectral : public Spectral {
    public:
    PartialMockSpectral(Config& config_, DiscretizationGrid& grid_)
    : Spectral(config_, grid_) {}
    using array2 = std::array<int, 2>;
    MOCK_METHOD(void, mechanical_response, (double Delta_t, int cell_start, int cell_end), (override));
    MOCK_METHOD(void, thermal_response, (double Delta_t, int cell_start, int cell_end), (override));
    MOCK_METHOD(void, mechanical_response2, (double Delta_t, array2& FEsolving_execIP, array2& FEsolving_execElem), (override));
  };

  gridSetup_init_grid(std::array<int, 3>{2,1,1});
  gridSetup_init_discretization();
  f_homogenization_init();

  PartialMockSpectral spectral(config, *mock_grid);
  spectral.homogenization_fetch_tensor_pointers();
  gridSetup_init_tensorfield(spectral, *mock_grid);

  spectral.wgt = 0.5;
  spectral.homogenization_dPdF->setValues({
   {{{{  159842489344.4433 ,  159972848994.13599 },
      { -1504219541.5565622, -1689836307.3836253 },
      {  3448398456.3962173, -2102190460.1164069 }},
     {{ -1504219541.556551 , -1689836307.3836384 },
      {  95515693258.175293,  91421799586.101868 },
      {  397176716.55049217, -1733315101.8327127 }},
     {{  3448398456.3962283, -2102190460.1164093 },
      {  397176716.55048251, -1733315101.8327119 },
      {  92351794310.386627,  96315328332.7677   }}},
    {{{ -1504219541.5565717, -1689836307.3836272 },
      {  32478211607.960251,  28384317935.887001 },
      {  397176716.55047393, -1733315101.832706  }},
     {{  32478211607.960377,  28384317935.886948 },
      {  3794034107.7043304,  3061285023.9171753 },
      { -1798204152.5553036, -674227858.25496078 }},
     {{  397176716.55048466, -1733315101.8327007 },
      { -1798204152.5553031, -674227858.25495911 },
      { -2289814566.1477509, -1371448716.5334778 }}},
    {{{  3448398456.3962212, -2102190460.1164107 },
      {  397176716.55047107, -1733315101.8327065 },
      {  29314312660.171581,  33277846682.552811 }},
     {{  397176716.5504818 , -1733315101.8327031 },
      { -1798204152.5553017, -674227858.25495911 },
      { -2289814566.1477551, -1371448716.5334837 }},
     {{  29314312660.171707,  33277846682.552757 },
      { -2289814566.1477566, -1371448716.5334852 },
      { -1650194303.8409281,  2776418318.3713531 }}}},
   {{{{ -1504219541.5565605, -1689836307.3836365 },
      {  32478211607.960388,  28384317935.886944 },
      {  397176716.55048269, -1733315101.8327076 }},
     {{  32478211607.960255,  28384317935.887012 },
      {  3794034107.7043309,  3061285023.9171615 },
      { -1798204152.5553055, -674227858.25496244 }},
     {{  397176716.55048376, -1733315101.8327024 },
      { -1798204152.5553002, -674227858.25496101 },
      { -2289814566.1477585, -1371448716.5334778 }}},
    {{{  95515693258.175308,  91421799586.101929 },
      {  3794034107.7043247,  3061285023.917181  },
      { -1798204152.5552964, -674227858.25495613 }},
     {{  3794034107.7043424,  3061285023.9171824 },
      {  156612696277.37164,  159441337877.85236 },
      {  2153860905.3987808, -1076600383.6635833 }},
     {{ -1798204152.5552955, -674227858.25496471 },
      {  2153860905.3987808, -1076600383.663578  },
      {  95581587377.458328,  96846839449.05127  }}},
    {{{  397176716.55048752, -1733315101.8327026 },
      { -1798204152.5553038, -674227858.25495815 },
      { -2289814566.1477599, -1371448716.5334854 }},
     {{ -1798204152.5553057, -674227858.25496149 },
      {  2153860905.3987532, -1076600383.6635823 },
      {  32544105727.243263,  33809357798.836369 }},
     {{ -2289814566.1477561, -1371448716.5334842 },
      {  32544105727.243397,  33809357798.836308 },
      { -2551037621.9492874,  2809915485.4962654 }}}},
   {{{{  3448398456.3962164, -2102190460.1164055 },
      {  397176716.55048019, -1733315101.8327076 },
      {  29314312660.171707,  33277846682.552742 }},
     {{  397176716.5504815 , -1733315101.8327029 },
      { -1798204152.5553007, -674227858.25496674 },
      { -2289814566.1477566, -1371448716.5334835 }},
     {{  29314312660.171577,  33277846682.552826 },
      { -2289814566.1477542, -1371448716.5334742 },
      { -1650194303.8409517,  2776418318.3713517 }}},
    {{{  397176716.5504837 , -1733315101.8327045 },
      { -1798204152.5553048, -674227858.25495839 },
      { -2289814566.1477623, -1371448716.5334854 }},
     {{ -1798204152.5553007, -674227858.25495863 },
      {  2153860905.3987808, -1076600383.6635818 },
      {  32544105727.243397,  33809357798.836304 }},
     {{ -2289814566.1477561, -1371448716.5334761 },
      {  32544105727.243271,  33809357798.836376 },
      { -2551037621.9492846,  2809915485.4962573 }}},
    {{{  92351794310.386642,  96315328332.767685 },
      { -2289814566.1477537, -1371448716.5334697 },
      { -1650194303.8409224,  2776418318.3713655 }},
     {{ -2289814566.1477404, -1371448716.5334749 },
      {  95581587377.458328,  96846839449.051208 },
      { -2551037621.9492702,  2809915485.4962602 }},
     {{ -1650194303.8409293,  2776418318.3713498 },
      { -2551037621.9492855,  2809915485.4962564 },
      {  159776595225.16037,  154547809131.18652 }}}}
  });

  Eigen::Tensor<double, 5> P(3, 3, 2, 1, 1);
  P.setValues({
   {{{{  4.6813928926783649e-310 }}, {{  4.6803066082175087e-310 }}},
    {{{  0                       }}, {{  4.9406564584124654e-324 }}},
    {{{  0                       }}, {{  0                      }}}},
   {{{{  0                       }}, {{  4.6803066083258079e-310 }}},
    {{{  9                       }}, {{  4.9406564584124654e-324 }}},
    {{{  6.9048496684497202e-310 }}, {{  0                      }}}},
   {{{{  0                       }}, {{  9.8813129168249309e-324 }}},
    {{{  131                     }}, {{  1.9762625833649862e-323 }}},
    {{{  4.680306608333713e-310  }}, {{  4.9406564584124654e-324 }}}}
  });
  Eigen::TensorMap<Eigen::Tensor<double, 5>> P_map(P.data(), 3, 3, mock_grid->cells[0], mock_grid->cells[1], mock_grid->cells[2]);

  Eigen::Tensor<double, 2> P_av(3, 3);
  P_av.setZero();

  Eigen::Tensor<double, 4> C_vol_avg(3, 3, 3, 3);
  C_vol_avg.setZero();

  Eigen::Tensor<double, 4> C_minmax_avg(3, 3, 3, 3);
  C_minmax_avg.setZero();

  Eigen::Tensor<double, 5> F(3, 3, 2, 1, 1);
  F.setValues({
   {{{{  1 }}, {{  1 }}}, {{{  0 }}, {{  0 }}}, {{{  0 }}, {{  0 }}}},
   {{{{  0 }}, {{  0 }}}, {{{  1 }}, {{  1 }}}, {{{  0 }}, {{  0 }}}},
   {{{{  0 }}, {{  0 }}}, {{{  0 }}, {{  0 }}}, {{{  1 }}, {{  1 }}}}
  });
  Eigen::TensorMap<Eigen::Tensor<double, 5>> F_map(F.data(), 3, 3, mock_grid->cells[0], mock_grid->cells[1], mock_grid->cells[2]);


  double delta_t = 0;

  Eigen::Tensor<double, 3> mech2_homogenization_P_mock_result(3, 3, 2);
  mech2_homogenization_P_mock_result.setValues({
   {{ -0.00012625698593137619,  6.7368932803888855e-05 },
    {  2.4478419622059409e-07,  9.2854840990643325e-06 },
    { -5.539774355740229e-06 , -1.4117250738593625e-06 }},
   {{  2.4478419622060764e-07,  9.2854840990643325e-06 },
    { -0.00012980115459804682,  6.0970263434762862e-05 },
    { -8.8118593343441355e-06, -4.1239652459827732e-07 }},
   {{ -5.5397743557402239e-06, -1.4117250738593625e-06 },
    { -8.8118593343441388e-06, -4.1239652459827563e-07 },
    { -0.00012997748173127732,  6.4678614891698315e-05 }}
  });
  EXPECT_CALL(spectral, mechanical_response(testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());
  EXPECT_CALL(spectral, thermal_response(testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());
  EXPECT_CALL(spectral, mechanical_response2(testing::_, testing::_, testing::_))
    .WillOnce([&](double Delta_t, 
                  std::array<int, 2>& FEsolving_execIP, 
                  std::array<int, 2>& FEsolving_execElem) {
        *spectral.homogenization_P = mech2_homogenization_P_mock_result;
        return;
    });

  Eigen::Tensor<double, 4> expected_C_vol_avg(3, 3, 3, 3);
  expected_C_vol_avg.setValues({
   {{{  159907669169.28964, -1597027924.4700937,  673103998.13990521 },
     { -1597027924.4700947,  93468746422.13858 , -668069192.64111018 },
     {  673103998.13990951, -668069192.64111471,  94333561321.577164 }},
    {{ -1597027924.4700994,  30431264771.923626, -668069192.64111602 },
     {  30431264771.92366 ,  3427659565.8107529, -1236216005.4051323 },
     { -668069192.64110804, -1236216005.4051311, -1830631641.3406143 }},
    {{  673103998.13990521, -668069192.64111769,  31296079671.362198 },
     { -668069192.64111066, -1236216005.4051304, -1830631641.3406196 },
     {  31296079671.362232, -1830631641.340621 ,  563112007.26521254 }}},
   {{{ -1597027924.4700985,  30431264771.923668, -668069192.64111245 },
     {  30431264771.923634,  3427659565.8107462, -1236216005.405134  },
     { -668069192.64110935, -1236216005.4051306, -1830631641.3406181 }},
    {{  93468746422.138611,  3427659565.8107529, -1236216005.4051263 },
     {  3427659565.8107624,  158027017077.612  ,  538630260.86759877 },
     { -1236216005.4051301,  538630260.86760139,  96214213413.254791 }},
    {{ -668069192.64110756, -1236216005.4051309, -1830631641.3406227 },
     { -1236216005.4051337,  538630260.86758542,  33176731763.039818 },
     { -1830631641.34062  ,  33176731763.039852,  129438931.773489   }}},
   {{{  673103998.13990545, -668069192.64111376,  31296079671.362225 },
     { -668069192.64111066, -1236216005.4051337, -1830631641.34062   },
     {  31296079671.362202, -1830631641.3406143,  563112007.26520002 }},
    {{ -668069192.64111042, -1236216005.4051316, -1830631641.3406239 },
     { -1236216005.4051297,  538630260.86759949,  33176731763.039848 },
     { -1830631641.3406162,  33176731763.039825,  129438931.77348638 }},
    {{  94333561321.577164, -1830631641.3406117,  563112007.2652216  },
     { -1830631641.3406076,  96214213413.254761,  129438931.77349496 },
     {  563112007.26521027,  129438931.77348542,  157162202178.17346 }}}
  });

  spectral.constitutive_response(P_map, P_av, C_vol_avg, C_minmax_avg, F_map, delta_t);
  EXPECT_TRUE(tensor_eq(C_vol_avg, expected_C_vol_avg));
}


// TEST(SpectralTestFFTRandom, BasicTest) {
//   MockDiscretization mock_discretization;
//   int cells_[] = {2, 1, 1};
//   double geom_size_[] = {2e-5, 1e-5, 1e-5};
//   MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
//   Spectral spectral(mock_grid);

//   ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
//   spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
//                        spectral.tensorField_real, spectral.tensorField_fourier, spectral.tensorField_fourier_fftw,
//                        FFTW_MEASURE, spectral.plan_tensor_forth, spectral.plan_tensor_back,
//                        "tensor");
//   mock_grid->cells1_tensor = cells1_fftw;
//   mock_grid->cells1_offset_tensor = cells1_offset;

//   fill_random(*spectral.tensorField_real);

//   spectral.tensorField_real->slice(Eigen::array<long, 5>({0, 0, mock_grid->cells[0], 0, 0}), 
//                                    Eigen::array<long, 5>({3, 3, mock_grid->cells0_reduced * 2 - mock_grid->cells[0], -1, -1})).setConstant(0);

//   auto tensorField_real_copy = *(spectral.tensorField_real);

//   fftw_mpi_execute_dft_r2c(spectral.plan_tensor_forth, spectral.tensorField_real->data(), spectral.tensorField_fourier_fftw);

//   std::complex<double> sum =   (*spectral.tensorField_fourier);
  // double tensorSumLocal = tensorField_real_copy.sum();
  // MPI_Allreduce(&tensorSumLocal, &tensorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // double avg = tensorSum / tensorField_fourier->dimension(2) / tensorField_fourier->dimension(3) / tensorField_fourier->dimension(4);
  // double epsilon = 1e-12;

  // if (world_rank == 0) {
  //   if (std::abs(avg - 1.0) > epsilon) {
  //     std::cerr << "Mismatch avg tensorField FFT <-> real" << std::endl;
  //     MPI_Abort(MPI_COMM_WORLD, 1);
  //   }
  // }

  // fftw_execute(plan_tensor_back);

  // // Set the specified range to zero
  // tensorField_real->slice(Eigen::array<long, 5>({0, 0, cells(1), 0, 0}), Eigen::array<long, 5>({3, 3, cells1Red * 2 - cells(1), -1, -1})) = Eigen::Tensor<double, 5>::Zero(3, 3, cells1Red * 2 - cells(1), grid.cells[1], cells2_fftw);

  // double max_difference = (*tensorField_real - tensorField_real_copy * wgt).abs().
// }


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}