#!/usr/bin/python3

import subprocess
import os
from pathlib import Path
import shutil
import concurrent.futures

files_to_copy = [
    'phase_damage.f90',
    'phase_thermal_source_externalheat.f90',
    'phase_mechanical_plastic_isotropic.f90',
    'polynomials.f90',
    'phase_thermal.f90',
    'result.f90',
    'homogenization_mechanical.f90',
    'discretization.f90',
    'misc.f90',
    'phase_mechanical_elastic.f90',
    'rotations.f90',
    'phase_mechanical_plastic.f90',
    'phase_mechanical_plastic_nonlocal.f90',
    'prec.f90',
    'phase_mechanical_eigen.f90',
    'geometry_plastic_nonlocal.f90',
    'tables.f90',
    'phase_mechanical_plastic_phenopowerlaw.f90',
    'signal.f90',
    'homogenization.f90',
    'material.f90',
    'homogenization_mechanical_RGC.f90',
    'homogenization_thermal_isotemperature.f90',
    'phase_mechanical_plastic_dislotwin.f90',
    'phase.f90',
    'math.f90',
    'HDF5_utilities.f90',
    'config.f90',
    'constants.f90',
    'YAML_types.f90',
    'system_routines.f90',
    'C_routines.c',
    'parallelization.f90',
    'tensor_printer.f90',
    'phase_mechanical_plastic_kinehardening.f90',
    'phase_mechanical_plastic_none.f90',
    'homogenization_mechanical_pass.f90',
    'IO.f90',
    'phase_thermal_source_dissipation.f90',
    'homogenization_damage.f90',
    'homogenization_damage_pass.f90',
    'LAPACK_interface.f90',
    'phase_mechanical.f90',
    'phase_damage_isobrittle.f90',
    'phase_mechanical_plastic_dislotungsten.f90',
    'phase_damage_anisobrittle.f90',
    'quit.f90',
    'homogenization_mechanical_isostrain.f90',
    'homogenization_thermal_pass.f90',
    'YAML_parse.f90',
    'homogenization_thermal.f90',
    'phase_mechanical_eigen_thermalexpansion.f90'
]


intel = True
intel_command = "source /opt/intel/oneapi/setvars.sh && export LD_LIBRARY_PATH=~/petsc/linux-intel/lib:$LD_LIBRARY_PATH &&"

# Define the directories where the command should be executed
dir1 = '/home/snkr/DAMASK_' # f90 damask
dir2 = '/home/snkr/DAMASK' # cpp damask
cmd = '-g mytest.vti -l tensionX.yaml -m material.yaml'
upper_fraction = 1

for filename in files_to_copy:
    src_file_path = os.path.join(dir2, 'src', filename)  # Full path to source file
    dst_file_path = os.path.join(dir1, 'src', filename)  # Full path to destination file
    
    # Check if the source file exists
    if os.path.exists(src_file_path):
        shutil.copy2(src_file_path, dst_file_path)  # copy2 preserves file metadata
    else:
        print(f"{src_file_path} does not exist!")


def execute_command(directory, command, capture_output=False):
    """Executes a command in the specified directory and returns the output."""
    try:
        # Change to the specified directory
        os.chdir(directory)
        
        # Execute the command
        result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
        # Check for errors
        if result.returncode != 0:
            print(f"Error executing command in {directory}: [{result.stderr}]")
        
        # Return success status and captured output (if any)
        return result.returncode == 0, (result.stdout if capture_output else "")
    
    except Exception as e:
        print(f"Error executing command in {directory}: {e}")
        return False, ""


def task1():
    print("Compiling original damask")
    success1, output1_compile = execute_command(dir1, f'make grid BUILD_TYPE=DEBUG', capture_output=True)
    print("Finished compiling, Running original damask")
    success2, output1_run = execute_command(Path(dir2, 'examples', 'grid'), f'{dir1}/bin/DAMASK_grid {cmd}', capture_output=True)
    print("Finished running original damask")
    return output1_compile, output1_run

def task2():
    print("Compiling cpp solver damask")
    success1, output2_compile = execute_command(dir2, f'make grid  BUILD_TYPE=DEBUG')
    print("Finished compiling, Running cpp solver damask")
    success2, output2_run = execute_command(Path(dir2, 'examples', 'grid'), f'{dir2}/bin/DAMASK_grid {cmd}', capture_output=True)

    # success2, output2_run = execute_command(Path(dir2), '~/solver_integrated/bin/DAMASK_grid', capture_output=True)
    print("Finished running cpp solver damask")
    return output2_compile, output2_run

# Use ThreadPoolExecutor to run tasks in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    future1 = executor.submit(task1)
    future2 = executor.submit(task2)
    
    # If you want to retrieve the results
    output1_compile, output1_run = future1.result()
    output2_compile, output2_run = future2.result()

# output2 = execute_command(dir2, 'make grid && ./bin/DAMASK_grid')
print("writing outputs...")
with open(Path(os.path.dirname(__file__),'output_damask_f.sh'), 'w') as f:
    f.write(output1_compile + '='*96 + output1_run)
with open(Path(os.path.dirname(__file__),'output_damask_cpp.sh'), 'w') as f:
    f.write(output2_compile + '\n' + '='*32 + '\n' + output2_run)
print("computing diff...")
# Compare the outputs and print the diff
if output1_compile is not None and output2_compile is not None:
    # If you want to find and print the exact differences
    import difflib
    differ = difflib.Differ()
    o1 = output1_run.splitlines()
    o2 = output2_run.splitlines()
    o1 = o1[:int(len(o1)/upper_fraction)]
    o2 = o2[:int(len(o2)/upper_fraction)]
    diff = differ.compare(o1, o2)
    print("len1", len(o1))
    print("len2", len(o2))
    print("saving")
    with open(Path(os.path.dirname(__file__), 'diff.patch'), 'w') as f:
        for line in diff:
            f.write(line + '\n')
    print("saved")

    