import damask
import numpy as np


cells = [1,2,1]
size = np.array(cells)*1e-5

N_grains = 2
seeds = damask.seeds.from_random(size,N_grains,cells)
grid = damask.Grid.from_Voronoi_tessellation(cells,size,seeds)
grid.save(f'test_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}')