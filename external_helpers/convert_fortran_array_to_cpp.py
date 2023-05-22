import numpy as np

def format_nested_array(numbers_str, dimensions, cat=False):
    if 'T' in numbers_str:
        numbers_str = numbers_str.replace('T', '1').replace('F', '0')
    is_complex = True if '(' in numbers_str else False
    numbers = [float(x) for x in numbers_str.split()] if not is_complex else \
              [complex(float(x.split(',')[0][1:]), float(x.split(',')[-1][:-1])) for x in numbers_str.split()]
    nl = '\n'
    numbers_array = np.array(numbers)
    reshaped_array = np.reshape(numbers_array, dimensions, order='F')
    max_len = (max([len(f'{i.real:.17g}') for i in np.ravel(numbers_array)]) if not is_complex else \
              max(max(len(f'{i.real:.17g}'), len(f'{i.imag:.17g}')) for i in np.ravel(numbers_array)))
    def format_array(arr, level, max_len, inner_first=False):
        indent = ' '+' '*level
        if level == len(dimensions) - 1:
            inner_str = []
            for x in arr:
                val = (f"c({'' if x.real<0 else ' '}{x.real:.17g}{' '*(max_len-len(f'{abs(x.real):.17g}')-1)}, "
                         f"{'' if x.imag<0 else ' '}{x.imag:.17g}{' '*(max_len-len(f'{abs(x.imag):.17g}')-1)})") if is_complex else \
                      f"{'' if x<0 else ' '}{x:.17g}{' '*(max_len-len(f'{abs(x):.17g}')-1)}"
                inner_str.append(val)
            return f'{{ {", ".join(inner_str)} }}'
        else:
            inner_str = f',{nl}{indent}'.join(
                [f'{format_array(x, level + 1, max_len, i == 0)}' for i, x in enumerate(arr)])
            if level == 0:
                return f'{{{nl}{indent}{inner_str}{nl}}}'
            else:
                return f'{{{nl+indent if not cat else ""}{inner_str}{nl+indent if not cat else ""}}}'

    formatted_output = format_array(reshaped_array, 0, max_len)
    return formatted_output