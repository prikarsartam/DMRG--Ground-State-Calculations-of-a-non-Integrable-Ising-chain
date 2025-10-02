import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite

import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

np.set_printoptions(precision=10, suppress=True, linewidth=100)

# __all__ = ['ThreeSpinIsing']

class ThreeSpinIsing(CouplingMPOModel):
    r"""Finite NNN Ising model.

    The Hamiltonian for three-spin Ising model:

    .. math ::
        H = - \sum_i [ J \sigma^x_{i} \sigma^x_{i+1} \sigma^x_{i+2} + h \sigma^z_i ]

    Parameters
    ----------
    model_params : dict
        Parameters for the model.
    J, h : float
        Coupling parameters.
    L : int
        Length of the chain.
    bc_MPS : string
        Boundary conditions for the MPS ('finite' or 'infinite').
    bc_x : string
        Boundary conditions in the x direction ('open' or 'periodic'). 
        Default is 'open' for finite chains and 'periodic' for infinite chains.
        Note that open boundary conditions are only implemented for finite chains.
    conserve : string
        Which quantum number to conserve ('Sz', 'N', or None). If 'best', it will be 'Sz' if h=0 and None otherwise.
    """
    def __init__(self, model_params):
        super().__init__(model_params)

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'Sz' if self.options.get('h', 0.) == 0 else None
        site = SpinHalfSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1., 'real_or_array'))
        h = np.asarray(model_params.get('h', 1., 'real_or_array'))

        self.add_onsite(-h, 0, 'Sigmaz')          # Onsite terms: -h * Sigmaz_i
        self.add_multi_coupling(-J, [('Sigmax', [0], 0), ('Sigmax', [1], 0), ('Sigmax', [2], 0)])   # Three-site terms: -J * Sigmax_i * Sigmax_{i+1} * Sigmax_{i+2}



# finite chain calculation of ground state and reduced density matrix
# default boundary : 'periodic', change to 'open' if needed

def fin_DMRG_GS_ThreeSpinIsing(L, h, boundary_condition = 'periodic'):
    # periodic boundary condition is necessary for infinity system calculations

    model_params__inf_3_blck = {
        'L': L,
        'J': 1.0,
        'h': h ,
        'conserve': None,
        'bc_MPS': 'finite',
        'bc_x': boundary_condition,
    }

    model = ThreeSpinIsing(model_params__inf_3_blck)

    psi = MPS.from_lat_product_state(model.lat, [['up']])

    dmrg_params = {
        'mixer': True,

        'trunc_params': {
            'chi_max': 500,
            'svd_min': 1.e-15,
            'trunc_cut': None,
        },
        # 'diag_method': 'lanczos',
        # 'max_E_err': 1.e-10,
        # 'max_S_err': 1.e-6,
        # 'N_sweeps_check': 5,
        'max_sweeps': 50,
        # 'E_tol_to_trunc': 0.05,
        # 'P_tol_to_trunc': 0.05,
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params) 
    fin_gs_energy_density, fin_gs = eng.run()
    return fin_gs_energy_density, fin_gs

# size = 12
# h_val = 1.0
# gs_E, gs = fin_DMRG_GS_ThreeSpinIsing(size, h_val)
# print("\n\nGround state energy density: ", gs_E)


def fin_DMRG_RDM_(L, h, subsystem_site_index_list):
    subsys_size = len(subsystem_site_index_list)
    _, fin_gs = fin_DMRG_GS_ThreeSpinIsing(L, h)
    return fin_gs.get_rho_segment(subsystem_site_index_list).to_ndarray().reshape(2**subsys_size,2**subsys_size)

def dh__fin_DMRG_RDM_(L, h, subsystem_site_index_list, dh=1e-6):
    rdm_ = fin_DMRG_GS_ThreeSpinIsing(L, h, subsystem_site_index_list)
    rdm_plus = fin_DMRG_GS_ThreeSpinIsing(L, h + dh, subsystem_site_index_list)
    dh__rdm__ = ( rdm_plus - rdm_ ) / dh
    return rdm_, dh__rdm__





# finite chain calculation of ground state and reduced density matrix
# only 'periodic' boundary condition is implemented because for 'open', 
# the system is not translationally invariant so the program will
# not halt at the critical point rendering unreliable results


def iDMRG_GS_(h):
    # periodic boundary condition is necessary for infinity system calculations

    model_params__inf_3_blck = {
        'L': 3,
        'J': 1.0,
        'h': h ,
        'conserve': None,
        'bc_MPS': 'infinite',
        'bc_x': 'periodic',
    }

    model = ThreeSpinIsing(model_params__inf_3_blck)

    psi = MPS.from_lat_product_state(model.lat, [['up']])

    dmrg_params = {
        'mixer': True,
        'trunc_params': {
            'chi_max': 50,
            'svd_min': 1.e-12,
            'trunc_cut': None,
        },
        # 'diag_method': 'lanczos',
        'max_E_err': 1.e-10,
        'max_S_err': 1.e-6,
        # 'N_sweeps_check': 5,
        'max_sweeps': 50,
        'E_tol_to_trunc': 0.05,
        'P_tol_to_trunc': 0.05,
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params) 
    inf_gs_energy_density, inf_gs = eng.run() 
    return inf_gs_energy_density, inf_gs


# gs_E_den, inf_gs = iDMRG_GS_(1.0)
# print("\n\nGround state energy density: ", gs_E_den)


def iDMRG_RDM_(h, subsystem_site_index_list):
    subsys_size = len(subsystem_site_index_list)
    _, inf_gs = iDMRG_GS_(h)
    return inf_gs.get_rho_segment(subsystem_site_index_list).to_ndarray().reshape(2**subsys_size,2**subsys_size)

def dh__iDMRG_RDM_(L, h, subsystem_site_index_list, dh=1e-6):
    rdm_ = iDMRG_RDM_(L, h, subsystem_site_index_list)
    rdm_plus = iDMRG_RDM_(L, h + dh, subsystem_site_index_list)
    dh__rdm__ = ( rdm_plus - rdm_ ) / dh
    return rdm_, dh__rdm__


def metric_response__ThreeSpinIsing(L, h, subsystem_site_index_list, dh=1e-6):
    # L = size and L = -1 for infinite system
    if L != -1:
        rdm_, dh_rdm_ = dh__fin_DMRG_RDM_(L, h, subsystem_site_index_list, dh)
    else:
        rdm_, dh_rdm_ = dh__iDMRG_RDM_(L, h, subsystem_site_index_list, dh)
    return (1/2) * np.trace(rdm_ @ dh_rdm_ @ dh_rdm_)
