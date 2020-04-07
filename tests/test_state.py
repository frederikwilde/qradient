from qradient.physical_components import State
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import scipy.sparse.linalg as slg


tol = 10.**-10
n = 5
s = State(n)
s.activate_center_matrix()
s.set_center_matrix(sp.identity(2**n))

def test_reset_vec():
    s.vec = np.random.rand(2**n)
    s.reset()
    v = np.zeros(2**n, dtype='complex')
    v[0] = 1.
    assert lg.norm(s.vec - v) < tol

def test_xrot_vec():
    s.load_xrots()
    s.reset()
    s.xrot(np.pi/2, 3)
    v = np.zeros(2**n, dtype='complex')
    v[0] = 1/np.sqrt(2)
    v[2] = -1j/np.sqrt(2)
    assert lg.norm(s.vec - v) < tol

def test_yrot_vec():
    s.load_yrots()
    s.reset()
    s.yrot(np.pi/2, 3)
    v = np.zeros(2**n, dtype='complex')
    v[0] = 1/np.sqrt(2)
    v[2] = 1/np.sqrt(2)
    assert lg.norm(s.vec - v) < tol

def test_zrot_vec():
    s.load_zrots()
    s.vec = 2.**(-.5*n) * np.ones(2**n, dtype='complex')
    s.zrot(np.pi, 2)
    v = 2.**(-.5*n) * np.ones(2**n, dtype='complex')
    for i in range(4, 32, 8):
        v[i:i+4] *= 1.j
    for i in range(0, 32, 8):
        v[i:i+4] *= -1.j
    assert lg.norm(s.vec - v) < tol

def test_cnot_ladder_vec():
    s.load_cnot_ladder()
    s.vec = np.zeros(2**n, dtype='complex')
    s.vec[2**(n-1)] = 1.  # the |1,0,0,0,0> state
    s.cnot_ladder(0)
    v = np.zeros(2**n, dtype='complex')
    v[24] = 1.  # the |1,1,0,0,0> state
    assert lg.norm(s.vec - v) < tol
    # other stacking
    s.vec = np.zeros(2**n, dtype='complex')
    s.vec[2**(n-1)] = 1.  # the |1,0,0,0,0> state
    s.cnot_ladder(1)
    v = np.zeros(2**n, dtype='complex')
    v[28] = 1.  # the |1,1,1,0,0> state
    assert lg.norm(s.vec - v) < tol

### CENTER MATRIX TESTS ###
def test_reset_center():
    s.center_matrix = sp.random(2**n, 2**n)
    s.reset()
    m = sp.identity(2**n)
    assert slg.norm(s.center_matrix - m) < tol

def test_xrot_center():
    pass
