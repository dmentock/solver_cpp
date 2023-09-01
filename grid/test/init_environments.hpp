#include <gtest/gtest.h>

#include <petsc.h>
#include <petscsys.h>


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