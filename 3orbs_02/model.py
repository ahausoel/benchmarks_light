import sys, os
sys.path.append(os.getcwd() + '/../common')
from util import *

from pytriqs.gf import *
from pytriqs.operators import c, c_dag, n
from pytriqs.operators.util.hamiltonians import h_int_kanamori
from itertools import product
from numpy import matrix, array, diag, eye
from numpy.linalg import inv

# ==== System Parameters ====
beta = 35.                       # Inverse temperature
mu = 2.5
eps = array([0.0, 0.1, -0.2])         # Impurity site energies
t = 0.0                         # Hopping between impurity sites

eps_bath = array([0.27, -0.04, 0.1, 0.27, -0.04, 0.1])  # Bath site energies
t_bath = 0.0                    # Hopping between bath sites

U = 0.00001                          # On-site interaction
V = 0.00001                         # Intersite interaction
J = 0.000001                        # Hunds coupling

spin_names = ['up', 'dn']
orb_names  = [0, 1, 2]
orb_bath_names  = [0, 1, 2, 3, 4, 5]

# Non-interacting impurity hamiltonian in matrix representation
h_0_mat = diag(eps - mu) - matrix([[0, t, t/2],
                                   [t, 0, t  ],
                                   [t/2, t, 0  ]])

# Bath hamiltonian in matrix representation
h_bath_mat = diag(eps_bath)

# Coupling matrix
V_mat = matrix([[1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4, 1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.2, 0.4]])

# ==== Local Hamiltonian ====
c_dag_vec = { s: matrix([[c_dag(s,o) for o in orb_names]]) for s in spin_names }
c_vec =     { s: matrix([[c(s,o)] for o in orb_names]) for s in spin_names }

h_0 = sum(c_dag_vec[s] * h_0_mat * c_vec[s] for s in spin_names)[0,0]

h_int = h_int_kanamori(spin_names, orb_names,
                        array([[0,      V-3*J, V-3*J  ],
                               [V-3*J, 0,      V-3*J  ],
                               [V-3*J, V-3*J, 0       ]]),    # Interaction for equal spins
                        array([[U,      U-2*J,  U-2*J   ],
                               [U-2*J,  U,      U-2*J   ],
                               [U-2*J,  U-2*J,  U       ]]),    # Interaction for opposite spins
                        J,True)

h_loc = h_0 + h_int

# ==== Bath & Coupling hamiltonian ====
orb_bath_names = ['b_' + str(o) for o in orb_bath_names]
c_dag_bath_vec = { s: matrix([[c_dag(s, o) for o in orb_bath_names]]) for s in spin_names }
c_bath_vec =     { s: matrix([[c(s, o)] for o in orb_bath_names]) for s in spin_names }

h_bath = sum(c_dag_bath_vec[s] * h_bath_mat * c_bath_vec[s] for s in spin_names)[0,0]
h_coup = sum(c_dag_vec[s] * V_mat * c_bath_vec[s] + c_dag_bath_vec[s] * V_mat.transpose() * c_vec[s] for s in spin_names)[0,0] # FIXME Adjoint

# ==== Total impurity hamiltonian ====
h_imp = h_loc + h_coup + h_bath

# ==== Green function structure ====
gf_struct = [ [s, orb_names] for s in spin_names ]

# ==== Hybridization Function ====
n_iw = int(10 * beta)
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
Delta = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)
# FIXME Delta['up'] << V_mat * inverse(iOmega_n - h_bath_mat) * V_mat.transpose()
for bl, iw in product(spin_names, iw_mesh):
    Delta[bl][iw] = V_mat * inv(iw.value * eye(len(orb_bath_names)) - h_bath_mat) * V_mat.transpose()

# ==== Non-Interacting Impurity Green function  ====
G0_iw = Delta.copy()
G0_iw['up'] << inverse(iOmega_n - h_0_mat - Delta['up']) # FIXME Should work for BlockGf
G0_iw['dn'] << inverse(iOmega_n - h_0_mat - Delta['dn'])
