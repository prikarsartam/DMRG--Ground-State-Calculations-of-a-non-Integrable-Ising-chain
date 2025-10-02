import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite

np.set_printoptions(precision=10, suppress=True, linewidth=100)

__all__ = ['ThreeSpinIsing']

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