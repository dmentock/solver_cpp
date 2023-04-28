// module to test basic interoperability between c++ and f90 programs. Linked with subroutines from stub_fortran_interface.f90

#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <array_matcher.h>

extern "C" {
void __fortran_testmodule_MOD_fortran_datatypes_test(
  int *int_num,
  double *double_num,
  int *arr1d, 
  int *arr2d, 
  int *arr3d
  );
}
class DatatypeTest : public ::testing::Test {};
TEST_F(DatatypeTest, TestDatatypes) {
  int int_num;
  double double_num;
  int arr1d[3];
  int arr2d[3][3];
  int arr3d[3][3][3];
  __fortran_testmodule_MOD_fortran_datatypes_test(
    &int_num,
    &double_num,
    &arr1d[0],
    &arr2d[0][0],
    &arr3d[0][0][0]);
  EXPECT_EQ(int_num, 1);
  //c++ double and .f90 real interoperability
  double double_error_margin_upper = 4.76838e-08;
  double double_error_margin_lower = 4.76837e-08;
  EXPECT_TRUE (fabs(double_num-2.3)<double_error_margin_upper);
  EXPECT_TRUE (fabs(double_num-2.3)>double_error_margin_lower);
  //row major vs col major
  int arr1d_[3] = {1,2,3};
  EXPECT_TRUE (std::equal(arr1d, arr1d + 3, arr1d_));
  int arr2d_[3][3] = {{1,4,7},{2,5,8},{3,6,9}};
  EXPECT_TRUE (std::equal(&arr2d[0][0], &arr2d[0][0] + 9, &arr2d_[0][0]));
  int arr3d_[3][3][3] = {{{1,10,19},{4,13,22},{7,16,25}},
                         {{2,11,20},{5,14,23},{8,17,26}},
                         {{3,12,21},{6,15,24},{9,18,27}}};
  EXPECT_TRUE (std::equal(&arr3d[0][0][0], &arr3d[0][0][0] + 27, &arr3d_[0][0][0]));
}

// test if calling of fortran functions in real and mocked interfaces works as expected
extern "C" {
  int __fortran_testmodule_MOD_fortran_interface_test_func(int* output_val);
}
class FortranFuncInterface {
  public:
    virtual int fortran_interface_test_func(int* output_val) {
      return __fortran_testmodule_MOD_fortran_interface_test_func(output_val);
    }
};
class MockFortranFuncInterface : public FortranFuncInterface {
public:
  MOCK_METHOD(int, fortran_interface_test_func, (int*), (override));
};
int testfunc_direct(FortranFuncInterface* func_interface) {
  int testoutput;
  func_interface->fortran_interface_test_func(&testoutput);
  return testoutput;
}
class InterfaceTest : public ::testing::Test {};
TEST_F(InterfaceTest, FortranFunctionSetsValues) {
  FortranFuncInterface func_interface;
  MockFortranFuncInterface mock_func_interface;
  ASSERT_EQ(testfunc_direct(&func_interface), 123);
  EXPECT_CALL(mock_func_interface, fortran_interface_test_func(testing::_))
      .WillOnce(::testing::DoAll(::testing::SetArgPointee<0>(456), ::testing::Return(0)));
  ASSERT_EQ(testfunc_direct(&mock_func_interface), 456);
  }

// test if scalar- and array values passed to fortran functions can be verified accordingly
class MockFortranFuncInterfacePointerArgs {
public:
    MOCK_METHOD(void, fortran_interface_test_func_pointerargs, (int*, int*));
};
void testfunc_pointer(MockFortranFuncInterfacePointerArgs* func_interface)
{
    int testscalar = 123;
    int testarray[3] = {3,5,7};
    func_interface->fortran_interface_test_func_pointerargs(&testscalar,&testarray[0]);
}
TEST_F(InterfaceTest, FortranFunctionIsCalledWithPointerValues)
{
    MockFortranFuncInterfacePointerArgs mock_func_interface;
    int expected_scalar = 123;
    std::vector<int> expected_array = {3, 5, 7};
    EXPECT_CALL(mock_func_interface, fortran_interface_test_func_pointerargs(
        testing::Pointee(expected_scalar),
        ArrayPointee(3, testing::ElementsAreArray(expected_array))
    ));
    testfunc_pointer(&mock_func_interface);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}