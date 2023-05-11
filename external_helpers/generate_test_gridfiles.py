import damask
import numpy as np
size = np.ones(3)*1e-5
size[0] *=2

cells = [2,1,1]
N_grains = 2
seeds = damask.seeds.from_random(size,N_grains,cells)
grid = damask.Grid.from_Voronoi_tessellation(cells,size,seeds)
grid.save(f'test_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}')