#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <cstdio>
#include <fstream>

#include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <fftw3-mpi.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#include "simple_grid_setup.hpp"
#include "init_environments.hpp"
#include "spectral/spectral.h"
#include "spectral/mech/utilities.h"

#include "test/array_matcher.h"
#include "tensor_operations.h"
#include "test/complex_concatenator.h"
#include "helper.h"



// class SpectralMockFFTW : public Spectral {
// public:
//   SpectralMockFFTW(DiscretizationGrid& grid_) : Spectral(grid_) {}
//   template <int Rank>
//   void set_up_fftw (ptrdiff_t& cells1_fftw, 
//                     ptrdiff_t& cells1_offset, 
//                     ptrdiff_t& cells2_fftw,
//                     int size,
//                     std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, Rank>>>& field_real,
//                     std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, Rank>>>& field_fourier,
//                     fftw_complex*& field_fourier_fftw,
//                     int fftw_planner_flag,
//                     fftw_plan& plan_forth, 
//                     fftw_plan& plan_back,
//                     const std::string& label) {
//     cells1_fftw = 1;
//     cells1_offset = 0;
//     std::cout << "mockymockymockymockymockymockymockymockymockymockymockymockymockymockymockymockymockymocky" << std::endl;
//   }
// };

TEST_F(SimpleGridSetup, MechUtilitiesTestInit) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  init_tensorfield(spectral, *mock_grid);

  Utilities mech_utilities(config, *mock_grid, spectral);

  config.num_grid.memory_efficient = 0;
  Eigen::DSizes<Eigen::DenseIndex, 7> expected_gamma_hat_dims(3, 3, 3, 3, 2, 1, 1);
  mech_utilities.init_utilities();
  ASSERT_EQ(mech_utilities.gamma_hat.dimensions(), expected_gamma_hat_dims);

  config.num_grid.memory_efficient = 1;
  Eigen::DSizes<Eigen::DenseIndex, 7> expected_gamma_hat_dims_mem_eff(3, 3, 3, 3, 1, 1, 1);
  mech_utilities.init_utilities();
  ASSERT_EQ(mech_utilities.gamma_hat.dimensions(), expected_gamma_hat_dims_mem_eff);

  // TODO: mock calls to set_up_fftw template function
}

TEST_F(SimpleGridSetup,MechUtilitiesTestInitDivergenceCorrection) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  mock_grid->geom_size = std::array<double, 3>{3,4,5};

  Utilities mech_utilities(config, *mock_grid, spectral);

  config.num_grid.divergence_correction = 0;
  mech_utilities.init_utilities();
  ASSERT_EQ(mock_grid->scaled_geom_size, mock_grid->geom_size);

  config.num_grid.divergence_correction = 1;
  mech_utilities.init_utilities();
  std::array<double, 3>expected_scaled_geom_size_1({0.75, 1, 1.25});
  ASSERT_EQ(mock_grid->scaled_geom_size, expected_scaled_geom_size_1);

  config.num_grid.divergence_correction = 2;
  mech_utilities.init_utilities();
  std::array<double, 3>expected_scaled_geom_size_2({0.6, 0.8, 1 });
  ASSERT_EQ(mock_grid->scaled_geom_size, expected_scaled_geom_size_2);
}

TEST_F(SimpleGridSetup, MechUtilitiesTestUpdateCoords) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);

  init_tensorfield(spectral, *mock_grid);
  init_vectorfield(spectral, *mock_grid);
  spectral.wgt = 0.5;

  Utilities mech_utilities(config, *mock_grid, spectral);

  spectral.xi2nd.resize(3, 2, 1, 1);
  spectral.xi2nd.setValues({
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  314159.2653589793) }}},
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  0               ) }}},
   {{{ c( 0               ,  0               ) }},
    {{ c( 0               ,  0               ) }}}
  });

  Eigen::Tensor<double, 5> F(3, 3, 2, 1, 1);
  F.setValues({
   {{{{  1  }}, {{  1  }}}, {{{  0  }}, {{  0  }}}, {{{  0  }}, {{  0  }}}}, 
   {{{{  0  }}, {{  0  }}}, {{{  1  }}, {{  1  }}}, {{{  0  }}, {{  0  }}}},
   {{{{  0  }}, {{  0  }}}, {{{  0  }}, {{  0  }}}, {{{  1  }}, {{  1  }}}}
  });

  Eigen::Tensor<double, 2> expected_x_n(3, 12);
  expected_x_n.setValues({
   {  0   ,  1e-05, 2e-05,  0    ,  1e-05,  2e-05,  0    ,  1e-05,  2e-05,  0    ,  1e-05,  2e-05 },
   {  0   ,  0   ,  0    ,  1e-05,  1e-05,  1e-05,  0    ,  0    ,  0    ,  1e-05,  1e-05,  1e-05 },
   {  0   ,  0   ,  0    ,  0    ,  0    ,  0    ,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05,  1e-05 }
  });

  Eigen::Tensor<double, 2> expected_x_p(3, 2);
  expected_x_p.setValues({
   {  5e-06                ,  1.5e-05               },
   {  5e-06                ,  5e-06                 },
   {  5e-06                ,  5e-06                 }
  });

  Eigen::Tensor<double, 2> x_n;
  Eigen::Tensor<double, 2> x_p;
  mech_utilities.update_coords(F, x_n, x_p);
  EXPECT_TRUE(tensor_eq(x_n, expected_x_n));
  EXPECT_TRUE(tensor_eq(x_p, expected_x_p));
}

TEST_F(SimpleGridSetup, TestUpdateGamma) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  spectral.wgt = 0.5;

  Utilities mech_utilities(config, *mock_grid, spectral);

  Eigen::Tensor<double, 4> C_min_max_avg(3,3,3,3);
  C_min_max_avg.setValues({
    {
      {{11.11,11.12,11.13},{11.21,11.22,11.23},{11.31,11.32,11.33}},
      {{12.11,12.12,12.13},{12.21,12.22,12.23},{12.31,12.32,12.33}},
      {{13.11,13.12,13.13},{13.21,13.22,13.23},{13.31,13.32,13.33}}
    },{
      {{21.11,21.12,21.13},{21.21,21.22,21.23},{21.31,21.32,21.33}},
      {{22.11,22.12,22.13},{22.21,22.22,22.23},{22.31,22.32,22.33}},
      {{23.11,23.12,23.13},{23.21,23.22,23.23},{23.31,23.32,23.33}}
    },{
      {{31.11,31.12,31.13},{31.21,31.22,31.23},{31.31,31.32,31.33}},
      {{32.11,32.12,32.13},{32.21,32.22,32.23},{32.31,32.32,32.33}},
      {{33.11,33.12,33.13},{33.21,33.22,33.23},{33.31,33.32,33.33}}
    }
  });

  spectral.xi1st.resize(3, 2, 1, 1);
  spectral.xi1st.setValues({
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}}
  });

  config.num_grid.memory_efficient = 0;

  mech_utilities.gamma_hat.resize(3, 3, 3, 3, 1, 1, 1);
  Eigen::Tensor<std::complex<double>, 7> expected_gamma_hat(3, 3, 3, 3, 1, 1, 1);
  expected_gamma_hat.setConstant(std::complex<double>(0.0, 0.0));
  mech_utilities.update_gamma(C_min_max_avg);
  EXPECT_TRUE(tensor_eq(mech_utilities.gamma_hat, expected_gamma_hat));
  // TODO: add mpi test with initialized instead of mocked discretization
  // TODO: find testcases that cause gamma fluctuation
}

TEST_F(SimpleGridSetup, MechUtilitiesTestForwardField) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  spectral.wgt = 0.5;

  Utilities mech_utilities(config, *mock_grid, spectral);

  Eigen::Tensor<double, 5> field_last_inc(3, 3, 2, 1, 1);
  field_last_inc.setValues({
   {{{{  1  }}, {{  1  }}}, {{{  0  }}, {{  0  }}}, {{{  0  }}, {{  0  }}}},
   {{{{  0  }}, {{  0  }}}, {{{  1  }}, {{  1  }}}, {{{  0  }}, {{  0  }}}},
   {{{{  0  }}, {{  0  }}}, {{{  0  }}, {{  0  }}}, {{{  1  }}, {{  1  }}}}
  });

  Eigen::Tensor<double, 5> rate(3, 3, 2, 1, 1);
  rate.setValues({
   {{{{  0.001 }}, {{  0.001 }}}, {{{  0    }}, {{  0    }}}, {{{  0    }}, {{  0    }}}},
   {{{{  0     }}, {{  0     }}}, {{{  0    }}, {{  0    }}}, {{{  0    }}, {{  0    }}}},
   {{{{  0     }}, {{  0     }}}, {{{  0    }}, {{  0    }}}, {{{  0    }}, {{  0    }}}}
  });

  Eigen::Matrix<double, 3, 3> aim;
  aim.resize(3,3);
  aim << 1.001,  0   ,  0,
         0    ,  1   ,  0,
         0    ,  0   ,  1;

  double delta_t = 1;

  Eigen::Tensor<double, 5> expected_forwarded_field(3, 3, 2, 1, 1);
  expected_forwarded_field.setValues({
   {{{{  1.001 }}, {{  1.001 }}}, {{{  0    }}, {{  0    }}}, {{{  0    }}, {{  0    }}}},
   {{{{  0     }}, {{  0     }}}, {{{  1    }}, {{  1    }}}, {{{  0    }}, {{  0    }}}},
   {{{{  0     }}, {{  0     }}}, {{{  0    }}, {{  0    }}}, {{{  1    }}, {{  1    }}}}
  });

  Eigen::Tensor<double, 5> forwarded_field;
  forwarded_field.resize(3, 3, 2, 1, 1);
  mech_utilities.forward_field(delta_t, field_last_inc, rate, forwarded_field, &aim);
  EXPECT_TRUE(tensor_eq(forwarded_field, expected_forwarded_field));
}

TEST_F(SimpleGridSetup, MechUtilitiesTestMaskedCompliance) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  spectral.wgt = 0.5;

  Utilities mech_utilities(config, *mock_grid, spectral);

  std::array<double, 4> rot_bc_q = {1, 0, 0, 0};

  Eigen::Matrix<bool, 3, 3> mask_stress;
  mask_stress << true, true, true,
                 true, false, true,
                 true, true, false;

  Eigen::Tensor<double, 4> C;
  C.resize(3, 3, 3, 3);
  C.setValues({
   {{{  76463759461.90884,  151900112.1841662,  322805058.0901647 },
     {  145203207.856026 ,  74853637761.91884, -16161985.30529766 },
     {  306508570.2876508, -16172746.25663342,  74884229223.97279 }},
    {{  167794786.7287251,  5879391483.530503,  357241099.9451544 },
     {  5800070200.321621, -1977287826.209463, -1267170893.386347 },
     {  351920157.5559037, -1268014599.408121,  1851670008.198951 }},
    {{  359725855.2681346,  357536024.9558614,  6303190154.013287 },
     {  351976944.5132964, -1309330548.645288,  1505678353.210385 },
     {  6222291390.206772,  1506680861.948623,  1062276510.103906 }}},
   {{{  160883440.8253435,  5800070200.321619,  351685998.4569044 },
     {  5806978147.957319, -1945900307.731439, -1237763536.71773  },
     {  347172258.3312932, -1250907294.230903,  1826688368.450823 }},
    {{  74465042081.00273, -1978809575.856601, -1341498320.237682 },
     { -1947401526.829771,  81626273459.48462, -191205839.5900816 },
     { -1324280760.189613, -192079117.0802204,  72378735826.48666 }},
    {{  529582.009655501 , -1251634942.691855,  1486958152.024444 },
     { -1222437187.907851, -183909321.4614324,  4291007454.157637 },
     {  1467873676.937728,  4293864486.382861,  192298910.5025663 }}},
   {{{  342955504.7459785,  352211297.3296106,  6222291390.20677  },
     {  347459470.21887  , -1292525848.256194,  1486353612.308071 },
     {  6227590002.440372,  1492054371.484368,  1060954077.437887 }},
    {{  529934.6152462172, -1252468304.587812,  1487948196.503479 },
     { -1235570740.85205 , -184777740.7936628,  4293864486.382858 },
     {  1473562131.828285,  4296723422.095912,  193172419.4300823 }},
    {{  74533963142.73592,  1812316078.539639,  1001231831.910945 },
     {  1787865378.801924,  72391368771.25739,  183875403.8789575 },
     {  1000692881.711163,  184743304.2786026,  81392648197.50052 }}}
  });

  Eigen::Tensor<double, 4> expected_masked_compliance;
  expected_masked_compliance.resize(3, 3, 3, 3);
  expected_masked_compliance.setValues({
   {{{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }},
    {{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }},
    {{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }}},
   {{{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }},
    {{  0                    ,  0                    ,  0                     },
     {  0                    ,  5.79644245938708e-11 ,  0                     },
     {  0                    ,  0                    , -5.154509489399949e-11 }},
    {{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }}},
   {{{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }},
    {{  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     },
     {  0                    ,  0                    ,  0                     }},
    {{  0                    ,  0                    ,  0                     },
     {  0                    , -5.155409154653233e-11,  0                     },
     {  0                    ,  0                    ,  5.813080269043599e-11 }}}
  });

  Eigen::Tensor<double, 4> masked_compliance;
  masked_compliance.resize(3, 3, 3, 3);

  mech_utilities.calculate_masked_compliance(C, rot_bc_q, mask_stress, masked_compliance);
  EXPECT_TRUE(tensor_eq(masked_compliance, expected_masked_compliance));
  // TODO: Find more understandable test setup
}

TEST_F(SimpleGridSetup, MechUtilitiesTestDivergenceRMS) {
  init_grid(std::array<int, 3>{4,1,1}); // use larger grid to enter if branch
  Spectral spectral(config, *mock_grid);
  init_tensorfield(spectral, *mock_grid);
  spectral.wgt = 0.25;

  Utilities mech_utilities(config, *mock_grid, spectral);

  Eigen::Tensor<double, 5> tensor_field(3, 3, 4, 1, 1);
  tensor_field.setValues({
   {{{{  905457362.58214402 }},
     {{  905457362.58214402 }},
     {{  905456063.79715848 }},
     {{  905456063.79715848 }}},
    {{{  1214652.0334358416 }},
     {{  1214652.0334358416 }},
     {{  1252727.4895472536 }},
     {{  1252727.4895472536 }}},
    {{{ -789863.92346561013 }},
     {{ -789863.92346561013 }},
     {{ -238036.79438631469 }},
     {{ -238036.79438631469 }}}},
   {{{{  431252.19617307186 }},
     {{  431252.19617307186 }},
     {{  429889.43205684423 }},
     {{  429889.43205684423 }}},
    {{{  614484.09468809969 }},
     {{  614484.09468809969 }},
     {{ -614486.2450618872  }},
     {{ -614486.2450618872  }}},
    {{{  142408.38695884281 }},
     {{  142408.38695884281 }},
     {{ -627007.52767248882 }},
     {{ -627007.52767248882 }}}},
   {{{{ -179357.22246642411 }},
     {{ -179357.22246642411 }},
     {{ -179243.6871252954  }},
     {{ -179243.6871252954  }}},
    {{{  142485.39282193387 }},
     {{  142485.39282193387 }},
     {{ -626885.2751634228  }},
     {{ -626885.2751634228  }}},
    {{{ -841958.71212144871 }},
     {{ -841958.71212144871 }},
     {{  841957.63553933613 }},
     {{  841957.63553933613 }}}}
  });
  spectral.xi1st.resize(3, 3, 1, 1);
  spectral.xi1st.setValues({
   {{{ c( 0                ,  0                 ) }},
    {{ c( 0                ,  157079.63267948964) }},
    {{ c( 0                ,  0                 ) }}},
   {{{ c( 0                ,  0                 ) }},
    {{ c( 0                ,  0                 ) }},
    {{ c( 0                ,  0                 ) }}},
   {{{ c( 0                ,  0                 ) }},
    {{ c( 0                ,  0                 ) }},
    {{ c( 0                ,  0                 ) }}}
  });

  double rms = mech_utilities.calculate_divergence_rms(tensor_field);
  ASSERT_EQ(rms, 1481.2323577332318);
  // TODO: Add test for diverging
}

TEST_F(SimpleGridSetup, MechUtilitiesTestGammaConvolution) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  init_tensorfield(spectral, *mock_grid);

  Utilities mech_utilities(config, *mock_grid, spectral);

  Eigen::Tensor<double, 5> field(3, 3, 2, 1, 1);
  field.setValues({
   {{{{  9235641.189939182  }},
     {{  9632014.409984397  }}},
    {{{ -228992.9056867184  }},
     {{ -137151.7288854052  }}},
    {{{ -165027.6813614937  }},
     {{  277655.7139282213  }}}},
   {{{{ -228992.9056867179  }},
     {{ -137151.7288854057  }}},
    {{{  9558636.645616129  }},
     {{  9685168.179161072  }}},
    {{{ -255116.5173901524  }},
     {{  281005.5981252654  }}}},
   {{{{ -165044.1841296307  }},
     {{  277683.4794996136  }}},
    {{{ -255142.0290418919  }},
     {{  281033.698685078   }}},
    {{{  15980056.25126295  }},
     {{  15457099.20758347  }}}}
  });

  Eigen::Tensor<double, 2> field_aim(3, 3);
  field_aim.setValues({
   {  3.577335445476857e-05,  0                    ,  0                     },
   {  0                    ,  3.972872444401787e-05,  0                     },
   {  0                    ,  0                    ,  0                     }
  });

  spectral.xi1st.resize(3, 2, 1, 1);
  spectral.xi1st.setValues({
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}}
  });

  mech_utilities.C_ref.resize(3, 3, 3, 3);
  mech_utilities.C_ref.setValues({
   {{{  319815338338.5793 , -3194055848.940187 ,  1346207996.27981   },
     { -3194055848.940189 ,  186937492844.2772 , -1336138385.28222   },
     {  1346207996.279819 , -1336138385.282229 ,  188667122643.1543  }},
    {{ -3194055848.940199 ,  60862529543.84725 , -1336138385.282232  },
     {  60862529543.84732 ,  6855319131.621506 , -2472432010.810265  },
     { -1336138385.282216 , -2472432010.810262 , -3661263282.681229  }},
    {{  1346207996.27981  , -1336138385.282235 ,  62592159342.7244   },
     { -1336138385.282221 , -2472432010.810261 , -3661263282.681239  },
     {  62592159342.72446 , -3661263282.681242 ,  1126224014.530425  }}},
   {{{ -3194055848.940197 ,  60862529543.84734 , -1336138385.282225  },
     {  60862529543.84727 ,  6855319131.621492 , -2472432010.810268  },
     { -1336138385.282219 , -2472432010.810261 , -3661263282.681236  }},
    {{  186937492844.2772 ,  6855319131.621506 , -2472432010.810253  },
     {  6855319131.621525 ,  316054034155.224  ,  1077260521.735198  },
     { -2472432010.81026  ,  1077260521.735203 ,  192428426826.5096  }},
    {{ -1336138385.282215 , -2472432010.810262 , -3661263282.681245  },
     { -2472432010.810267 ,  1077260521.735171 ,  66353463526.07964  },
     { -3661263282.68124  ,  66353463526.0797  ,  258877863.546978   }}},
   {{{  1346207996.279811 , -1336138385.282228 ,  62592159342.72445  },
     { -1336138385.282221 , -2472432010.810267 , -3661263282.68124   },
     {  62592159342.7244  , -3661263282.681229 ,  1126224014.5304    }},
    {{ -1336138385.282221 , -2472432010.810263 , -3661263282.681248  },
     { -2472432010.810259 ,  1077260521.735199 ,  66353463526.0797   },
     { -3661263282.681232 ,  66353463526.07965 ,  258877863.5469728  }},
    {{  188667122643.1543 , -3661263282.681223 ,  1126224014.530443  },
     { -3661263282.681215 ,  192428426826.5095 ,  258877863.5469899  },
     {  1126224014.530421 ,  258877863.5469708 ,  314324404356.3469  }}}
  });

  Eigen::Tensor<double, 5> expected_gamma_field(3, 3, 2, 1, 1);
  expected_gamma_field.setValues({
   {{{{  3.577335445476857e-05 }},
     {{  3.577335445476857e-05 }}},
    {{{  0                     }},
     {{  0                     }}},
    {{{  0                     }},
     {{  0                     }}}},
   {{{{  0                     }},
     {{  0                     }}},
    {{{  3.972872444401787e-05 }},
     {{  3.972872444401787e-05 }}},
    {{{  0                     }},
     {{  0                     }}}},
   {{{{  0                     }},
     {{  0                     }}},
    {{{  0                     }},
     {{  0                     }}},
    {{{  0                     }},
     {{  0                     }}}}
  });

  mech_utilities.gamma_hat.resize(3,3,3,3,mock_grid->cells0_reduced,mock_grid->cells[2],mock_grid->cells1_tensor);
  mech_utilities.gamma_hat.setZero();

  config.num_grid.memory_efficient = 0;
  Eigen::Tensor<double, 5> gamma_field0 = mech_utilities.gamma_convolution(field, field_aim);
  EXPECT_TRUE(tensor_eq(gamma_field0, expected_gamma_field));

  config.num_grid.memory_efficient = 1;
  Eigen::Tensor<double, 5> gamma_field1 = mech_utilities.gamma_convolution(field, field_aim);
  EXPECT_TRUE(tensor_eq(gamma_field1, expected_gamma_field));
  //TODO: add test where det of A is large enough to enter if branch for memory_efficient=1
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}