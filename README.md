# Tensor-Network-based-Ground-State-Calculations-of-a-non-Integrable-Ising-chain
Tensor Network implementation of Three-Spin Ising model discussed in ArXiv: [2509.22515](https://arxiv.org/abs/2509.22515) using TenPy

The module contains the following functions

1. Definition of the Three-Spin Ising Hamiltonian $$H = - \sum_{i=1}^N ( \ J \ \sigma^x_i \sigma^x_{i+1} + h \  \sigma^z_i \ )$$
by inheriting the `CouplingMPOModel` from `tenpy.models.model`.
2. Determination of the ground state for finite (periodic+open boundary) and infinite (periodic boundary only) chains by optimizing a Tensor Network, namely a Matrix Product State (MPS) using Density Matrix Renormalization Group (DMRG) method,
3. Once ground (or any other) state is determined the reduced density matrices of a subsystem can be calculated by prescribing the array of indices for the subsystem,
4. Finally the calculation of the metric response of quantum relative entropy is implemented, as discussed in ArXiv: [2509.22515](https://arxiv.org/abs/2509.22515).
