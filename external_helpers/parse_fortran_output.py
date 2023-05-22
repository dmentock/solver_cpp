import convert_fortran_array_to_cpp as cf
import os

cat = True
indent = 2

def generate_cpp_tensor_def(item):
    vals = item.split('vals:')[-1]
    type = 'complex' if '(' in vals else 'double'
    name = item.split("  ")[0].strip()
    shape_str = item.split(name)[-1].split('vals:')[0]
    shape = [int(x) for x in shape_str.split()]
    nested_vals = cf.format_nested_array(vals, shape, cat = cat)
    return f"""
Eigen::Tensor<{'std::complex<double>' if type=='complex' else 'double'}, {len(shape)}> {name}({', '.join([str(i) for i in shape])});
{name}.setValues({nested_vals});"""

with open(os.path.join(os.path.dirname(__file__), 'fortran_output.txt'), 'r') as f:
    fortran_output = f.read()

for item in fortran_output.split('\n'):
    cpp_tensor_def = generate_cpp_tensor_def(item)
    print(cpp_tensor_def.replace('\n', f'\n{" "*indent}'))