from ngsolve import *
from os import environ
import numpy as np

rank = environ.get('LINHYP_RANK', 1)

mesh = Mesh(f'data/unit_square_{rank}.vol')
fes = L2(mesh, order=5, all_dofs_together=True)

Draw(mesh)

gfu = GridFunction(fes)
gfu.Set(0, definedon=VOL)
gfu.vec.FV().NumPy()[:] = np.loadtxt(f"data/cpp_concentration_{rank}.out")

Draw(gfu, autoscale=False, min=0, max=1)
