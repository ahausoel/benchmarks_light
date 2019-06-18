import sys, os
sys.path.append(os.getcwd() + '/../common')
from util import *

from pytriqs.gf import Gf, MeshImFreq, iOmega_n, inverse, MeshBrillouinZone, MeshProduct
from pytriqs.lattice import BravaisLattice, BrillouinZone
from pytriqs.operators import c, c_dag, n
from pytriqs.operators.util import h_int_kanamori, U_matrix_kanamori
from itertools import product
from numpy import matrix, array, diag, pi
import numpy.linalg as linalg

from tight_binding_model import *

# ==== System Parameters ====
beta = 10.                       # Inverse temperature
mu = 0.                         # Chemical potential

U=2
J=0.5
#U = 2.3                         # Density-density interaction
#J = 0.4                         # Hunds coupling
#U = 0.0000000001
#J = 0.0000000001
SOC = 0.3                       # Spin-orbit coupling

n_iw = int(100 * beta)           # The number of positive Matsubara frequencies
n_k = 16                        # The number of k-points per dimension

spin_names = ['up', 'dn']       # The spins
orb_names = [0, 1, 2]           # The orbitals
idx_lst = list(range(len(spin_names) * len(orb_names)))
gf_struct = [('bl', idx_lst)]
#print 'gf_struct', gf_struct
#exit()

TBL = tight_binding_model(lambda_soc=SOC)   # The Tight-Binding Lattice
TBL.bz = BrillouinZone(TBL.bl)
n_idx = len(idx_lst)

# ==== Local Hamiltonian ====
c_dag_vec = matrix([[c_dag('bl', idx) for idx in idx_lst]])
c_vec =     matrix([[c('bl', idx)] for idx in idx_lst])

h_0_mat = TBL._hop[(0,0,0)]
h_0 = (c_dag_vec * h_0_mat * c_vec)[0,0]

print 'h_0'
print  h_0_mat
#exit()

Umat, Upmat = U_matrix_kanamori(len(orb_names), U_int=U, J_hund=J)
#print 'Umat', Umat
#print 'Upmat', Upmat
op_map = { (s,o): ('bl',i) for i, (s,o) in enumerate(product(spin_names, orb_names)) }
#print 'op_map', op_map
#print ' '
#for s,o in product(spin_names, orb_names):
  #print 's,o', s,o
#print '--> orbital running fastest'
#exit()
h_int = h_int_kanamori(spin_names, orb_names, Umat, Upmat, J, off_diag=True, map_operator_structure=op_map)
h_loc = h_0 + h_int

print 'h_int', h_int
#exit()

# ==== Non-Interacting Impurity Green function  ====
iw_mesh = MeshImFreq(beta, 'Fermion', n_iw)
k_mesh = MeshBrillouinZone(TBL.bz, n_k)
k_iw_mesh = MeshProduct(k_mesh, iw_mesh)

#G0_k_iw = BlockGf(mesh=k_iw_mesh, gf_struct=gf_struct)
G0_iw = BlockGf(mesh=iw_mesh, gf_struct=gf_struct)

iw_vec = array([iw.value * np.eye(n_idx) for iw in iw_mesh])
k_vec = array([k.value for k in k_mesh])
e_k_vec = TBL.hopping(k_vec.T / 2. / pi).transpose(2, 0, 1)
mu_mat = mu * np.eye(n_idx)

#print 'type(iw_vec) ', type(iw_vec) 
#print 'type(mu_mat) ', type(mu_mat) 
#print 'type(e_k_vec) ', type(e_k_vec) 

#G0_k_iw['bl'].data[:] = linalg.inv(iw_vec[None,...] + mu_mat[None,None,...] - e_k_vec[::,None,...])
#G0_iw['bl'].data[:] = np.sum(G0_k_iw['bl'].data[:], axis=0) / len(k_mesh)

#print 'G0_iw["bl"].data[:]', G0_iw['bl'].data[:].shape
#print 'iw_vec.shape', iw_vec.shape
#print 'mu_mat.shape', mu_mat.shape
#print 'e_k_vec.shape', e_k_vec.shape

n_k_tot = e_k_vec.shape[0]
#print 'n_k_tot', n_k_tot

n_iw_tot = iw_vec.shape[0]
#print 'n_iw_tot', n_iw_tot

for k in range(0,n_k_tot):
    #print 'k', k
    G0_iw['bl'].data[:] += linalg.inv(iw_vec + mu_mat[np.newaxis,:,:] - e_k_vec[k, np.newaxis, :,:])

G0_iw['bl'].data[:] /= len(k_mesh)

# ==== Hybridization Function ====
Delta = G0_iw.copy()
Delta['bl'] << iOmega_n + mu_mat - h_0_mat - inverse(G0_iw['bl'])
