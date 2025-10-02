
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

from ThreeSpinIsingHamiltonian import ThreeSpinIsing


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

    dmrg_params_balanced = {
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

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params_balanced) 
    fin_gs_energy_density, fin_gs = eng.run() # the main work; modifies psi in place
    return fin_gs_energy_density, fin_gs

def fin_DMRG_RDM_(L, h, subsystem_site_index_list):
    subsys_size = len(subsystem_site_index_list)
    _, fin_gs = fin_DMRG_GS_ThreeSpinIsing(L, h)
    return fin_gs.get_rho_segment(subsystem_site_index_list).to_ndarray().reshape(2**subsys_size,2**subsys_size)

def dh__fin_DMRG_RDM_(L, h, subsystem_site_index_list, dh=1e-6):
    return ( fin_DMRG_GS_ThreeSpinIsing(L, h + dh, subsystem_site_index_list) - fin_DMRG_GS_ThreeSpinIsing(L, h - dh, subsystem_site_index_list) )/(2*dh)


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

    # dmrg_params_balanced = {
    #     'mixer': True,
    #     'trunc_params': {
    #         'chi_max': 50,
    #         'svd_min': 1.e-12,
    #         'trunc_cut': None,
    #     },
    #     # 'diag_method': 'lanczos',
    #     'max_E_err': 1.e-10,
    #     'max_S_err': 1.e-6,
    #     # 'N_sweeps_check': 5,
    #     'max_sweeps': 50,
    #     'E_tol_to_trunc': 0.05,
    #     'P_tol_to_trunc': 0.05,
    # }

    dmrg_params = {
        'mixer': True,
        'chi_list': {
            0: 50,     # Initial exploration
            5: 100,    # Build up slowly  
            15: 200,   # Increase systematically
            25: 500,   # High accuracy regime
            40: 1000,  # Maximum precision
            60: 2000   # Research-grade accuracy
        },
        'trunc_params': {
            'svd_min': 1.e-15,    # Machine precision
            'trunc_cut': None,    # Let chi_max dominate
        },
        'max_sweeps': 50,
        'verbose': 1,
}

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params) 
    inf_gs_energy_density, inf_gs = eng.run() # the main work; modifies psi in place
    return inf_gs_energy_density, inf_gs

def iDMRG_RDM_(h, subsystem_site_index_list):
    subsys_size = len(subsystem_site_index_list)
    _, inf_gs = iDMRG_GS_(h)
    return inf_gs.get_rho_segment(subsystem_site_index_list).to_ndarray().reshape(2**subsys_size,2**subsys_size)


# gs_E_den, inf_gs = iDMRG_GS_(1.0)
# print("\n\nGround state energy density: ", gs_E_den)




def metric_response__ThreeSpinIsing(L, h, subsystem_site_index_list, dh=1e-6):
    rdm_, dh_rdm_ = dh__fin_DMRG_RDM_(L, h, subsystem_site_index_list, dh)
    return (1/2) * np.trace(rdm_ @ dh_rdm_ @ dh_rdm_)