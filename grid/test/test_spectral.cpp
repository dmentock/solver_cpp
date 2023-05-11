#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array_matcher.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <fftw3-mpi.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#include "mock_discretized_grid.hpp"
#include "spectral/spectral.h"
#include "test/array_matcher.h"
#include "tensor_operations.h"
#include "test/complex_concatenator.h"
#include "helper.h"

class PetscMpiEnv : public ::testing::Environment {
protected:
  PetscErrorCode ierr;
public:
    virtual void SetUp() {
        int argc = 0;
        char **argv = NULL;
    ierr = PetscInitialize(&argc, &argv, (char *)0,"PETSc help message.");
    ASSERT_EQ(ierr, 0) << "Error initializing PETSc.";
  }
  virtual void TearDown() override {
    ierr = PetscFinalize();
    ASSERT_EQ(ierr, 0) << "Error finalizing PETSc.";
    }
};

class SpectralSetup : public ::testing::Test {
  protected:
    PetscErrorCode ierr;
  void SetUp() override {
    int argc = 0;
    char **argv = NULL;
    ierr = PetscInitialize(&argc, &argv, (char *)0,"PETSc help message.");
    ASSERT_EQ(ierr, 0) << "Error initializing PETSc.";
  }
  void TearDown() override {
    ierr = PetscFinalize();
    ASSERT_EQ(ierr, 0) << "Error finalizing PETSc.";
  }
};

class MinimalGridSetup : public ::testing::Test {
protected:
  std::unique_ptr<MockDiscretizedGrid> mock_grid;
public:
  void SetUp() {
    MockDiscretization mock_discretization;
    int cells_[] = {2, 1, 1};
    double geom_size_[] = {2e-5, 1e-5, 1e-5};
    mock_grid.reset(new MockDiscretizedGrid(mock_discretization, &cells_[0], &geom_size_[0]));
  }
  void init_tensorfield(Spectral& spectral, MockDiscretizedGrid& mock_grid){
    ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
    spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
                       spectral.tensorField_real, spectral.tensorField_fourier, spectral.tensorField_fourier_fftw,
                       FFTW_MEASURE, spectral.plan_tensor_forth, spectral.plan_tensor_back,
                       "tensor");
    mock_grid.cells1_tensor = cells1_fftw;
    mock_grid.cells1_offset_tensor = cells1_offset;
  }
  void init_vectorfield(Spectral& spectral, MockDiscretizedGrid& mock_grid){
    ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
    spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
                       spectral.vectorField_real, spectral.vectorField_fourier, spectral.vectorField_fourier_fftw,
                       FFTW_MEASURE, spectral.plan_vector_forth, spectral.plan_vector_back,
                       "vector");
  }
};
TEST_F(MinimalGridSetup,SpectralTestInit) {
  Spectral spectral(*mock_grid);

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
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  314159.2653589793) }}},
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  0               ) }}},
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  0               ) }}}
  });

  spectral.init();
  EXPECT_TRUE(tensor_eq(spectral.xi1st, expected_xi1st));
  EXPECT_TRUE(tensor_eq(spectral.xi2nd, expected_xi2nd));

  Eigen::DSizes<Eigen::DenseIndex, 7> expected_gamma_hat_dims(3, 3, 3, 3, 2, 1, 1);
  ASSERT_EQ(spectral.gamma_hat.dimensions(), expected_gamma_hat_dims);
}

// TEST_F(SpectralSetup, TestUpdateCoordsInit) {
//   MockDiscretization mock_discretization;
//   int cells_[] = {2, 1, 1};
//   double geom_size_[] = {2e-5, 1e-5, 1e-5};
//   MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
//   PartialMockSpectral spectral(mock_grid);

//   std::array<std::complex<double>, 3> res;
//   for (int i = 0; i < 3; ++i) res[i] = std::complex<double>(1,0);
//   EXPECT_CALL(spectral, get_freq_derivative(testing::_))
//     .WillRepeatedly(testing::DoAll(testing::Return(res)));

//   double expected_ip_coords_[3][12] = {
//     {
//         0, 1e-05, 2e-05,     0, 1e-05, 2e-05,     0, 1e-05, 2e-05,     0, 1e-05, 2e-05
//     }, {
//         0,     0,     0, 1e-05, 1e-05, 1e-05,     0,     0,     0, 1e-05, 1e-05, 1e-05
//     }, {
//         0,     0,     0,     0,     0,     0, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05
//     }
//   };
//   Eigen::array<Eigen::Index, 2> dims_ip_coords = {3, 12};
//   Eigen::Tensor<double, 2> expected_ip_coords = array_to_eigen_tensor<double, 2>(&expected_ip_coords_[0][0], dims_ip_coords);
//   EXPECT_CALL(mock_discretization, set_node_coords(testing::_))
//   .WillOnce(testing::Invoke([&](Eigen::Tensor<double, 2>* ip_coords){
//       bool eq = tensor_eq<double, 2>(*ip_coords, expected_ip_coords);
//       EXPECT_TRUE(eq);
//   })); 

//   double expected_node_coords_[3][2] = {
//     {
//       5e-06, 1.5e-05
//     }, {
//       5e-06, 1.000005
//     }, {
//       1.000005,   5e-06
//   }
//   };
//   Eigen::array<Eigen::Index, 2> dims_node_coords = {3, 2};
//   Eigen::Tensor<double, 2> expected_node_coords = array_to_eigen_tensor<double, 2>(&expected_node_coords_[0][0], dims_node_coords);

//   EXPECT_CALL(mock_discretization, set_ip_coords(testing::_))
//   .WillOnce(testing::Invoke([&](Eigen::Tensor<double, 2>* node_coords){
//       bool eq = tensor_eq<double, 2>(*node_coords, expected_node_coords);
//       EXPECT_TRUE(eq);
//   })); 

//   Eigen::Tensor<double, 5> F(3, 3, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]);
//   Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
//   for (int i = 0; i < 3; ++i)
//   {
//     for (int j = 0; j < 3; ++j)
//     {
//         F.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
//                 Eigen::array<Eigen::Index, 5>({1, 1, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]}))
//                 .setConstant(math_I3(i, j));
//     }
//   }
//   spectral.init(); // TODO: recreate only the required initializations in test constructor
//   spectral.update_coords(F);
// }


TEST_F(MinimalGridSetup, SpectralTestUpdateCoords) {
  Spectral spectral(*mock_grid);
  init_tensorfield(spectral, *mock_grid);
  init_vectorfield(spectral, *mock_grid);

  spectral.xi2nd.resize(3, 2, 1, 1);
  spectral.xi2nd.setValues({
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  314159.2653589793) }}},
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  0               ) }}},
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  0               ) }}}
  });

  spectral.wgt = 0.5;

  Eigen::Tensor<double, 2> expected_x_n(3, 12);
  expected_x_n.setValues({
   {  1.158890000000004e-310,  1.158890000000004e-310,  1.158890000000004e-310,  1.158890000000004e-310,  
      1.158890000000004e-310,  1e-05                 ,  1.158890000000004e-310,  1e-05                 ,  
      1.158950000000012e-310,  1.158950000000012e-310,  1e-05                 ,  1e-05                   },
   {  1e-05                 ,  1e-05                 ,  1e-05                 ,  1e-05                 ,  
      1.158890000000004e-310,  1e-05                 ,  1.158890000000004e-310,  1e-05                 ,  
      1.158950000000012e-310,  1.158950000000012e-310,  1e-05                 ,  1e-05                   },
   {  2e-05                 ,  2e-05                 ,  2e-05                 ,  2e-05                 ,  
      1.158890000000004e-310,  1e-05                 ,  1.158890000000004e-310,  1e-05                 ,  
      1.158950000000012e-310,  1.158950000000012e-310,  1e-05      ,  1e-05                              }
  });

  Eigen::Tensor<double, 2> expected_x_p(3, 2);
  expected_x_p.setValues({
   {   5e-06,  5e-06  },
   {  1.5e-05, 5e-06  },
   {   5e-06,  5e-06  }
  });

  Eigen::Tensor<double, 5> F(3, 3, mock_grid->cells[0], mock_grid->cells[1], mock_grid->cells[2]);
  Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j){
        F.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
                Eigen::array<Eigen::Index, 5>({1, 1, mock_grid->cells[0], mock_grid->cells[1], mock_grid->cells[2]}))
                .setConstant(math_I3(i, j));
    }
  }
  Eigen::Tensor<double, 2> x_n(3, 12);
  Eigen::Tensor<double, 2> x_p(3, 2);
  spectral.update_coords(F, x_n, x_p);
  EXPECT_TRUE(tensor_eq(x_n, expected_x_n));
  EXPECT_TRUE(tensor_eq(x_p, expected_x_p));
}

// TODO: add mpi test with initialized instead of mocked discretization
// TEST_F(SpectralSetup, TestUpdateGamma) {
//   MockDiscretization mock_discretization;
//   int cells_[] = {2, 1, 1};
//   double geom_size_[] = {2e-5, 1e-5, 1e-5};
//   MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
//   PartialMockSpectral spectral(mock_grid);

//   std::array<std::complex<double>, 3> res;
//   for (int i = 0; i < 3; ++i) res[i] = std::complex<double>(1,0);
//   EXPECT_CALL(spectral, get_freq_derivative(testing::_))
//     .WillRepeatedly(testing::DoAll(testing::Return(res)));

//   Eigen::Tensor<double, 4> C_min_max_avg(3,3,3,3);
//   C_min_max_avg.setValues({
//     {
//       {{11.11,11.12,11.13},{11.21,11.22,11.23},{11.31,11.32,11.33}},
//       {{12.11,12.12,12.13},{12.21,12.22,12.23},{12.31,12.32,12.33}},
//       {{13.11,13.12,13.13},{13.21,13.22,13.23},{13.31,13.32,13.33}}
//     },{
//       {{21.11,21.12,21.13},{21.21,21.22,21.23},{21.31,21.32,21.33}},
//       {{22.11,22.12,22.13},{22.21,22.22,22.23},{22.31,22.32,22.33}},
//       {{23.11,23.12,23.13},{23.21,23.22,23.23},{23.31,23.32,23.33}}
//     },{
//       {{31.11,31.12,31.13},{31.21,31.22,31.23},{31.31,31.32,31.33}},
//       {{32.11,32.12,32.13},{32.21,32.22,32.23},{32.31,32.32,32.33}},
//       {{33.11,33.12,33.13},{33.21,33.22,33.23},{33.31,33.32,33.33}}
//     }
//   });
//   spectral.init(); // TODO: recreate only the required initializations in test constructor
//   spectral.update_gamma(C_min_max_avg);
//   // print_tensor(&spectral.gamma_hat);
// }

TEST_F(SpectralSetup, TestUpdateGammaReal) {
  MockDiscretization mock_discretization;
  int cells_[] = {2, 1, 1};
  double geom_size_[] = {2e-5, 1e-5, 1e-5};
  MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
  Spectral spectral(mock_grid);
  spectral.wgt = 4.1666666666666664e-002;

  Eigen::Tensor<double, 4> C_min_max_avg(3,3,3,3);
  C_min_max_avg.setValues({
{
  {
    { 1.0959355524918582e+11, 3.3379178619552708e+08, 9.9655985396867371e+08 },
    { 3.3379178619552606e+08, 5.9373830941431534e+10, -1.8163978776946735e+08 },
    { 9.9655985396868014e+08, -1.8163978776946974e+08, 5.8602613809382538e+10 }
    },
  {
    { 3.3379178619553375e+08, 2.7303830941431511e+10, -1.8163978776947090e+08 },
    { 2.7303830941431534e+10, -1.0453692167466900e+09, -8.4651024104992294e+08 },
    { -1.8163978776947141e+08, -8.4651024104991937e+08, 7.1157743055114186e+08 }
    },
  {
    { 9.9655985396867180e+08, -1.8163978776947096e+08, 2.6532613809382538e+10 },
    { -1.8163978776947045e+08, -8.4651024104991782e+08, 7.1157743055114245e+08 },
    { 2.6532613809382553e+10, 7.1157743055114353e+08, -1.5004961291872242e+08 }
    }
  },
{
  {
    { 3.3379178619552892e+08, 2.7303830941431534e+10, -1.8163978776947221e+08 },
    { 2.7303830941431511e+10, -1.0453692167466853e+09, -8.4651024104992151e+08 },
    { -1.8163978776946959e+08, -8.4651024104992199e+08, 7.1157743055114293e+08 }
    },
  {
    { 5.9373830941431519e+10, -1.0453692167466898e+09, -8.4651024104991913e+08 },
    { -1.0453692167466860e+09, 1.0889425461969577e+11, -5.3278316695396471e+08 },
    { -8.4651024104991531e+08, -5.3278316695396245e+08, 5.9301914438872627e+10 }
    },
  {
    { -1.8163978776946834e+08, -8.4651024104992294e+08, 7.1157743055114436e+08 },
    { -8.4651024104992390e+08, -5.3278316695396757e+08, 2.7231914438872620e+10 },
    { 7.1157743055114436e+08, 2.7231914438872643e+10, 7.1442295472345102e+08 }
    }
  },
{
  {
    { 9.9655985396867156e+08, -1.8163978776947078e+08, 2.6532613809382553e+10 },
    { -1.8163978776946959e+08, -8.4651024104992568e+08, 7.1157743055114770e+08 },
    { 2.6532613809382519e+10, 7.1157743055114686e+08, -1.5004961291873169e+08 }
    },
  {
    { -1.8163978776946813e+08, -8.4651024104991841e+08, 7.1157743055114532e+08 },
    { -8.4651024104992223e+08, -5.3278316695396149e+08, 2.7231914438872643e+10 },
    { 7.1157743055114579e+08, 2.7231914438872608e+10, 7.1442295472345114e+08 }
    },
  {
    { 5.8602613809382538e+10, 7.1157743055114031e+08, -1.5004961291871765e+08 },
    { 7.1157743055114365e+08, 5.9301914438872627e+10, 7.1442295472345245e+08 },
    { -1.5004961291872787e+08, 7.1442295472344446e+08, 1.0966547175174469e+11 }
    }
  }
});
  spectral.init(); // TODO: recreate only the required initializations in test constructor
  spectral.update_gamma(C_min_max_avg); // TODO: assert values that are not all 0s
} 

// TEST_F(SpectralSetup, TestConstitutiveResponse) {
//   MockDiscretizedGrid grid;
//   Spectral spectral(grid);

//   int cells[3] = {2,3,4};
//   Eigen::Tensor<double, 5> P(3, 3, cells[0], cells[1], cells[2]);
//   Eigen::Tensor<double, 2> P_av(3, 3);
//   Eigen::Tensor<double, 4> C_volAvg(3, 3, 3, 3);
//   Eigen::Tensor<double, 4> C_minMaxAvg(3, 3, 3, 3);
//   Eigen::Tensor<double, 5> F(3, 3, cells[0], cells[1], cells[2]);
//   Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
//   for (int i = 0; i < 3; ++i)
//   {
//       for (int j = 0; j < 3; ++j)
//       {
//           F.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
//                   Eigen::array<Eigen::Index, 5>({1, 1, cells[0], cells[1], cells[2]}))
//               .setConstant(math_I3(i, j));
//       }
//   }
//   spectral.constitutive_response(P,
//                                   P_av,
//                                   C_volAvg,
//                                   C_minMaxAvg,
//                                   F,
//                                   0);
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}