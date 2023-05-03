import os
import yaml
import numpy as np

def format_nested_array(numbers_str, dimensions, cat=False):
    numbers = [float(x) for x in numbers_str.split()]
    nl = '\n'
    # Convert the list of numbers to a NumPy array
    numbers_array = np.array(numbers)

    # Reshape the array according to the given dimensions
    reshaped_array = np.reshape(numbers_array, dimensions, order='F')

    def format_array(arr, level, inner_first=False):
        indent = '  ' * level
        if level == len(dimensions) - 1:
            inner_str = ', '.join([f'{x:.16e}' for x in arr])
            return f'{{ {inner_str} }}'
        else:
            inner_str = f',\n{indent}'.join([f'{format_array(x, level + 1, i == 0)}' for i, x in enumerate(arr)])
            extra_space = ' ' if inner_first else ''
            return f'{{{nl+indent+extra_space if not cat else ""}{inner_str}{nl+indent if not cat else ""}}}'

    formatted_output = format_array(reshaped_array, 0)
    return formatted_output
with open(os.path.join(os.path.dirname(__file__), 'arr.yaml'), 'r') as f:
    arr_def = yaml.safe_load(f)

print(format_nested_array(arr_def['numbers'], arr_def['dims'], cat = arr_def.get('cat',False)))
