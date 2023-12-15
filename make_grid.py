import damask
import numpy as np

cells = [3,3,3]
size = np.ones(3)*1e-5
N_grains = 20

seeds = damask.seeds.from_random(size,N_grains,cells)
grid = damask.Grid.from_Voronoi_tessellation(cells,size,seeds)
grid.save(f'examples/grid/mytest.vti')
# grid.save(f'examples/grid/{cells[0]}x{cells[1]}x{cells[2]}.vti')
# grid

# material = np.zeros(grid,dtype=int)
# a = damask.Grid(material=material,size=grid)
# a.save(f'examples/grid/{grid[0]}x{grid[1]}x{grid[2]}.vti')
