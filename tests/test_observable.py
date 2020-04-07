import h5py
import numpy.linalg as lg
import numpy as np
from qradient.physical_components import Observable

tol = 10.**-10
def test_observable():
    x = np.array([None, 0.57646442, 0.07721182+0.84690635, 0.00243825, None])
    y = np.array([0.94714062+0.83249181, 0.07809328, 0.97936267, None, None])
    z = np.array([0.24611538, 0.8118797, 0.88232787, 0.27580387, None])
    zz = np.full((5, 5), None)
    zz[0, 1] = 0.52195169
    zz[2, 3] = 0.9098885
    zz[0, 2] = 0.97371475
    zz[1, 3] = 0.89770763
    with h5py.File('tests/test_observable.hdf5', 'r') as f:
        observable = Observable({'x': x, 'y': y, 'z': z, 'zz': zz})
        m = observable.matrix.asformat('array')
        assert lg.norm(m - f.get('matrix')[()]) < tol
